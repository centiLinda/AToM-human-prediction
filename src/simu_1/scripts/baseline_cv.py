#!/usr/bin/env python3

import rospy
from helpers.msg import Point4D, Point4DArray

''' 
Constant Velocity Baseline
'''

x1_initial, y1_initial, vx1_initial, vy1_initial = 0.0, 0.0, 0.0, 0.0
x1_initial_previous, y1_initial_previous, vx1_initial_previous, vy1_initial_previous = 0.0, 0.0, 0.0, 0.0
human_updated = False
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

def predict_cv():
    global x1_initial_previous, y1_initial_previous, vx1_initial_previous, vy1_initial_previous
    
    num_steps = 11 # include current position
    dt = 1.0

    predicted_trajectory = Point4DArray()
    for i in range(num_steps):
        point = Point4D()
        point.x = x1_initial_previous + i * dt * vx1_initial_previous
        point.y = y1_initial_previous + i * dt * vy1_initial_previous
        point.vx = vx1_initial_previous
        point.vy = vy1_initial_previous
        predicted_trajectory.points.append(point)
    
    return predicted_trajectory

def main():
    global run_solver, human_updated
    
    rospy.init_node('prediction_cv', anonymous=True)
    sub_h = rospy.Subscriber('/observed_human', Point4DArray, updateHumanCallback)
    pub_h = rospy.Publisher('/predicted_human', Point4DArray, queue_size=10)
    
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
            predicted_trajectory = predict_cv()
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