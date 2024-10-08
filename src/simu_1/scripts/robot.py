#!/usr/bin/env python3

import rospy
from helpers.msg import Point4D, Point4DArray 

#NOTE add robot dynamics & kill-switch here
def robot_plan_callback(msg):
    global action, received_plan

    action.x = msg.points[0].x
    action.y = msg.points[0].y
    action.vx = msg.points[0].vx
    action.vy = msg.points[0].vy
    received_plan = True

if __name__ == '__main__':
    try:
        rospy.init_node('robot')
        action = Point4D()
        received_plan = False
        trigger_processed = False

        pub = rospy.Publisher('/current_robot', Point4D, queue_size=10, latch=True)
        rospy.Subscriber('/robot_plan', Point4DArray, robot_plan_callback)
        rospy.sleep(0.1)

        # Initial state
        init_pos = Point4D()
        init_pos.x = rospy.get_param('/x2_start')
        init_pos.y = rospy.get_param('/y2_start')
        init_pos.vx = -1.0 #TODO better way to set initial vel
        init_pos.vy = 0
        pub.publish(init_pos)

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
