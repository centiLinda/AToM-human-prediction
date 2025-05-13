#!/usr/bin/env python3

import rospy
from helpers.msg import Point4D, Point4DArray 
import numpy as np

#NOTE add robot dynamics here
def robot_plan_callback(msg):
    global action, received_plan, current_human

    current_robot = np.array((msg.points[0].x, msg.points[0].y))
    human_threshold = 0.5
    wall_threshold = 0.2

    dist_to_human = np.linalg.norm(current_robot - current_human)
    if dist_to_human <= human_threshold:
        rospy.loginfo("###### Bump into human! ######")
        received_plan = True
        return

    x_pos = current_robot[0]
    y_pos = current_robot[1]
    lower_wall_y_max = 1.75
    wall_x_min = 2.0
    wall_x_max = 3.0
    too_close_to_lower_wall = (x_pos >= (wall_x_min - wall_threshold)) and (x_pos <= wall_x_max) and (y_pos <= (lower_wall_y_max + wall_threshold))
    if too_close_to_lower_wall:
        rospy.loginfo("# Bump into wall! #")
        received_plan = True
        return
    
    # turn 'move backwards' into 'wait'
    y_vel = msg.points[0].vy
    if (x_pos < 2) and (y_vel < 0):
        rospy.loginfo("# Freezed! #")
        received_plan = True
        return

    action.x = msg.points[0].x
    action.y = msg.points[0].y
    action.vx = msg.points[0].vx
    action.vy = msg.points[0].vy
    received_plan = True

def human_position_callback(msg):
    global current_human
    current_human = np.array((msg.points[-1].x, msg.points[-1].y))

if __name__ == '__main__':
    try:
        rospy.init_node('robot')
        action = Point4D()
        received_plan = False
        trigger_processed = False

        current_human = np.zeros(2)

        pub = rospy.Publisher('/current_robot', Point4D, queue_size=10, latch=True)
        rospy.Subscriber('/robot_plan', Point4DArray, robot_plan_callback)
        rospy.Subscriber('/observed_human', Point4DArray, human_position_callback)
        rospy.sleep(0.1)

        # Initial position
        init_pos = Point4D()
        init_pos.x = rospy.get_param('/x2_start', 10.0)
        init_pos.y = rospy.get_param('/y2_start', 5.0)
        init_pos.vx = 0.0 #TODO better way to set initial vel
        init_pos.vy = 0
        pub.publish(init_pos)

        action.x = init_pos.x
        action.y = init_pos.y

        rate = rospy.Rate(10) # check for trigger more frequently

        while not rospy.is_shutdown():
            trigger = rospy.get_param('/trigger', False)
            if trigger and not trigger_processed and received_plan:
                pub.publish(action)
                received_plan = False
                trigger_processed = True
                current_ready = rospy.get_param('/nodes_ready')
                rospy.set_param('/nodes_ready', current_ready + 1)                
            elif not trigger:
                trigger_processed = False
        
            rate.sleep()
        
    except rospy.ROSInterruptException:
        pass
