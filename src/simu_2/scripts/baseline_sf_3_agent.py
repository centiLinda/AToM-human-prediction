#!/usr/bin/env python3

import rospy
from helpers.msg import Point4D, Point4DArray, Point4DTwoArray, Point4DThreeArray
import numpy as np
import pysocialforce as psf

''' 
Social Force Baseline, adapted from https://github.com/yuxiang-gao/PySocialForce
'''

x1_initial, y1_initial, vx1_initial, vy1_initial = 0.0, 0.0, 0.0, 0.0
x1_initial_previous, y1_initial_previous, vx1_initial_previous, vy1_initial_previous = 0.0, 0.0, 0.0, 0.0
x2_initial, y2_initial, vx2_initial, vy2_initial = 0.0, 0.0, 0.0, 0.0
x2_initial_previous, y2_initial_previous, vx2_initial_previous, vy2_initial_previous = 0.0, 0.0, 0.0, 0.0
x3_initial, y3_initial, vx3_initial, vy3_initial = 0.0, 0.0, 0.0, 0.0
goals = np.zeros(6) # x1, y1, x2, y2
human_updated = False
robot_updated = False
run_solver = False

def updateHumanCallback(human_Received):
    global x1_initial, y1_initial, vx1_initial, vy1_initial, human_updated
    global x1_initial_previous, y1_initial_previous, vx1_initial_previous, vy1_initial_previous
    global x2_initial, y2_initial, vx2_initial, vy2_initial
    global x2_initial_previous, y2_initial_previous, vx2_initial_previous, vy2_initial_previous

    latest_1 = human_Received.agent1_traj.points[-1]
    x1_initial = latest_1.x
    y1_initial = latest_1.y
    vx1_initial = latest_1.vx
    vy1_initial = latest_1.vy

    latest_2 = human_Received.agent2_traj.points[-1]
    x2_initial = latest_2.x
    y2_initial = latest_2.y
    vx2_initial = latest_2.vx
    vy2_initial = latest_2.vy

    if len(human_Received.agent1_traj.points) > 1:
        second_latest_1 = human_Received.agent1_traj.points[-2]
        x1_initial_previous = second_latest_1.x
        y1_initial_previous = second_latest_1.y
        vx1_initial_previous = second_latest_1.vx
        vy1_initial_previous = second_latest_1.vy

        second_latest_2 = human_Received.agent2_traj.points[-2]
        x2_initial_previous = second_latest_2.x
        y2_initial_previous = second_latest_2.y
        vx2_initial_previous = second_latest_2.vx
        vy2_initial_previous = second_latest_2.vy

        human_updated = True

def updateRobotCallback(robot_Received):
    global x3_initial, y3_initial, vx3_initial, vy3_initial, robot_updated
    x3_initial = robot_Received.x
    y3_initial = robot_Received.y
    vx3_initial = robot_Received.vx
    vy3_initial = robot_Received.vy
    robot_updated = True

def predict_cv():
    global x1_initial_previous, y1_initial_previous, vx1_initial_previous, vy1_initial_previous
    global x2_initial_previous, y2_initial_previous, vx2_initial_previous, vy2_initial_previous
    global x3_initial, y3_initial, vx3_initial, vy3_initial
    global goals
    # prediction is base on previous, current is used for UKF update in AToM

    num_steps = 11 # include current position

    # (px, py, vx, vy, gx, gy)
    initial_state = np.array([[x1_initial_previous, y1_initial_previous, vx1_initial_previous, vy1_initial_previous, goals[0], goals[1]],
                              [x2_initial_previous, y2_initial_previous, vx2_initial_previous, vy2_initial_previous, goals[2], goals[3]],                              
                              [x3_initial, y3_initial, vx3_initial, vy3_initial, goals[4], goals[5]] 
                              ])
    groups = [[0], [1], [2]]
    obs = None

    s = psf.Simulator(initial_state,
                      groups=groups,
                      obstacles=obs,
                      config_file="PATH_TO_YOUR_WS/src/simu_2/scripts/config/corridor_3_agent_sf_config.toml")
    s.step(10)

    output, _ = s.get_states() # (step+1, 3, 7)
    x = output[:, 0, 0] # (step+1,) include current position
    y = output[:, 0, 1]
    vx = np.append(np.diff(x), np.diff(x)[-1])
    vy = np.append(np.diff(y), np.diff(y)[-1])
    predicted_trajectory_1 = Point4DArray()
    for i in range(num_steps):
        point = Point4D()
        point.x = x[i]
        point.y = y[i]
        point.vx = vx[i]
        point.vy = vy[i]
        predicted_trajectory_1.points.append(point)

    x = output[:, 1, 0] # (step+1,) include current position
    y = output[:, 1, 1]
    vx = np.append(np.diff(x), np.diff(x)[-1])
    vy = np.append(np.diff(y), np.diff(y)[-1])
    predicted_trajectory_2 = Point4DArray()
    for i in range(num_steps):
        point = Point4D()
        point.x = x[i]
        point.y = y[i]
        point.vx = vx[i]
        point.vy = vy[i]
        predicted_trajectory_2.points.append(point)
    
    return predicted_trajectory_1, predicted_trajectory_2

def main():
    global run_solver, human_updated, goals
    
    rospy.init_node('prediction_cv', anonymous=True)
    sub_h = rospy.Subscriber('/observed_human', Point4DTwoArray, updateHumanCallback)
    sub_r = rospy.Subscriber('/current_robot', Point4D, updateRobotCallback)
    pub_h = rospy.Publisher('/predicted_human', Point4DArray, queue_size=10)
    pub_all = rospy.Publisher('/game_results', Point4DThreeArray, queue_size=10)

    goals[0] = rospy.get_param('/x1_goal', 0.0)
    goals[1] = rospy.get_param('/y1_goal', 0.0)
    goals[2] = rospy.get_param('/x2_goal', 0.0)
    goals[3] = rospy.get_param('/y2_goal', 0.0)
    goals[4] = rospy.get_param('/x3_goal', 0.0)
    goals[5] = rospy.get_param('/y3_goal', 0.0)
    
    trigger = False
    trigger_processed = False
    rate = rospy.Rate(10)  # 10Hz
    
    while not rospy.is_shutdown():
        trigger = rospy.get_param('/trigger', False)
        
        if trigger and human_updated and not trigger_processed:
            run_solver = True
            human_updated = False
            trigger_processed = True
        elif not trigger:
            trigger_processed = False
            run_solver = False
        else:
            run_solver = False
        
        if run_solver:
            predicted_trajectory_1, predicted_trajectory_2 = predict_cv()
            pub_h.publish(predicted_trajectory_1)

            # dummy robot prediction
            zero_trajectory = Point4DArray()
            for _ in range(len(predicted_trajectory_1.points)):
                zero_point = Point4D()
                zero_point.x = 0.0
                zero_point.y = 0.0
                zero_point.vx = 0.0
                zero_point.vy = 0.0
                zero_trajectory.points.append(zero_point)
            msg_total = Point4DThreeArray()
            msg_total.agent1_traj = predicted_trajectory_1
            msg_total.agent2_traj = predicted_trajectory_2
            msg_total.agent3_traj = zero_trajectory
            pub_all.publish(msg_total)

            run_solver = False
            current_ready = rospy.get_param('/nodes_ready')
            rospy.set_param('/nodes_ready', current_ready + 1)
            
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass