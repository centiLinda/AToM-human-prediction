#!/usr/bin/env python3

import rospy
from helpers.msg import Point4D, Point4DArray, Point4DTwoArray, Point4DThreeArray

''' 
Constant Velocity Baseline
'''

x1_initial, y1_initial, vx1_initial, vy1_initial = 0.0, 0.0, 0.0, 0.0
x1_initial_previous, y1_initial_previous, vx1_initial_previous, vy1_initial_previous = 0.0, 0.0, 0.0, 0.0
x2_initial, y2_initial, vx2_initial, vy2_initial = 0.0, 0.0, 0.0, 0.0
x2_initial_previous, y2_initial_previous, vx2_initial_previous, vy2_initial_previous = 0.0, 0.0, 0.0, 0.0
human_updated = False
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

def predict_cv():
    global x1_initial_previous, y1_initial_previous, vx1_initial_previous, vy1_initial_previous
    global x2_initial_previous, y2_initial_previous, vx2_initial_previous, vy2_initial_previous
    
    num_steps = 11 # include current position
    dt = 1.0

    predicted_trajectory_1 = Point4DArray()
    predicted_trajectory_2 = Point4DArray()
    for i in range(num_steps):
        point = Point4D()
        point.x = x1_initial_previous + i * dt * vx1_initial_previous
        point.y = y1_initial_previous + i * dt * vy1_initial_previous
        point.vx = vx1_initial_previous
        point.vy = vy1_initial_previous
        predicted_trajectory_1.points.append(point)

        point = Point4D()
        point.x = x2_initial_previous + i * dt * vx2_initial_previous
        point.y = y2_initial_previous + i * dt * vy2_initial_previous
        point.vx = vx2_initial_previous
        point.vy = vy2_initial_previous
        predicted_trajectory_2.points.append(point)
    
    return predicted_trajectory_1, predicted_trajectory_2

def main():
    global run_solver, human_updated
    
    rospy.init_node('prediction_cv', anonymous=True)
    sub_h = rospy.Subscriber('/observed_human', Point4DTwoArray, updateHumanCallback)
    pub_h = rospy.Publisher('/predicted_human', Point4DArray, queue_size=10)
    pub_all = rospy.Publisher('/game_results', Point4DThreeArray, queue_size=10)
    
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