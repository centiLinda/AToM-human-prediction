#!/usr/bin/env python3

import rospy
from helpers.msg import Point4D, Point4DArray
import numpy as np
import pysocialforce as psf

''' 
Social Force Baseline, adapted from https://github.com/yuxiang-gao/PySocialForce
'''

x1_initial, y1_initial, vx1_initial, vy1_initial = 0.0, 0.0, 0.0, 0.0
x1_initial_previous, y1_initial_previous, vx1_initial_previous, vy1_initial_previous = 0.0, 0.0, 0.0, 0.0
x2_initial, y2_initial, vx2_initial, vy2_initial = 0.0, 0.0, 0.0, 0.0
goals = np.zeros(4) # x1, y1, x2, y2
human_updated = False
robot_updated = False
run_solver = False

def updateHumanCallback(human_Received):
    global x1_initial, y1_initial, vx1_initial, vy1_initial, human_updated
    global x1_initial_previous, y1_initial_previous, vx1_initial_previous, vy1_initial_previous

    latest = human_Received.points[-1]
    x1_initial = latest.x
    y1_initial = latest.y
    vx1_initial = latest.vx
    vy1_initial = latest.vy

    if len(human_Received.points) > 1:
        second_latest = human_Received.points[-2]
        x1_initial_previous = second_latest.x
        y1_initial_previous = second_latest.y
        vx1_initial_previous = second_latest.vx
        vy1_initial_previous = second_latest.vy
        human_updated = True

def updateRobotCallback(robot_Received):
    global x2_initial, y2_initial, vx2_initial, vy2_initial, robot_updated
    x2_initial = robot_Received.x
    y2_initial = robot_Received.y
    vx2_initial = robot_Received.vx
    vy2_initial = robot_Received.vy
    robot_updated = True

def predict_sf():
    global x1_initial_previous, y1_initial_previous, vx1_initial_previous, vy1_initial_previous
    global x2_initial, y2_initial, vx2_initial, vy2_initial
    global goals
    # prediction is base on previous, current is used for UKF update in AToM

    num_steps = 11 # include current position

    # (px, py, vx, vy, gx, gy)
    initial_state = np.array([[x1_initial_previous, y1_initial_previous, vx1_initial_previous, vy1_initial_previous, goals[0], goals[1]],
                              [x2_initial, y2_initial, vx2_initial, vy2_initial, goals[2], goals[3]] 
                              ])
    groups = [[0], [1]]
    obs = None

    s = psf.Simulator(initial_state,
                      groups=groups,
                      obstacles=obs,
                      config_file="PATH_TO_YOUR_WS/src/simu_1/scripts/config/simple_sf_config.toml")
    s.step(10)

    output, _ = s.get_states() # (step+1, 2, 7)
    x = output[:, 0, 0] # (step+1,) include current position
    y = output[:, 0, 1]
    vx = np.append(np.diff(x), np.diff(x)[-1])
    vy = np.append(np.diff(y), np.diff(y)[-1])

    predicted_trajectory = Point4DArray()
    for i in range(num_steps):
        point = Point4D()
        point.x = x[i]
        point.y = y[i]
        point.vx = vx[i]
        point.vy = vy[i]
        predicted_trajectory.points.append(point)
    
    return predicted_trajectory

def main():
    global run_solver, human_updated, goals
    
    rospy.init_node('prediction_sf', anonymous=True)
    sub_h = rospy.Subscriber('/observed_human', Point4DArray, updateHumanCallback)
    sub_r = rospy.Subscriber('/current_robot', Point4D, updateRobotCallback)
    pub_h = rospy.Publisher('/predicted_human', Point4DArray, queue_size=10)

    goals[0] = rospy.get_param('/x1_goal', 0.0)
    goals[1] = rospy.get_param('/y1_goal', 0.0)
    goals[2] = rospy.get_param('/x2_goal', 0.0)
    goals[3] = rospy.get_param('/y2_goal', 0.0)
    
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
            predicted_trajectory = predict_sf()
            pub_h.publish(predicted_trajectory)

            run_solver = False
            current_ready = rospy.get_param('/nodes_ready')
            rospy.set_param('/nodes_ready', current_ready + 1)
            
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass