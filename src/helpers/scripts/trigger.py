#!/usr/bin/env python3

import rospy

total_nodes = 4 # human, mpc, robot, ukf

def trigger():
    rospy.init_node('trigger')
    rospy.set_param('/trigger', False)
    rospy.set_param('/nodes_ready', 0)

    while not rospy.is_shutdown():
        input("\n---press 'Enter' to continue---") # wait for keyboard
        # directly kill with Ctrl+C causes hanging, because it stucks at this input()
        # press Enter first to exit input(), as an easy workaround

        rospy.set_param('/trigger', True)
        
        while not rospy.is_shutdown() and rospy.get_param('/nodes_ready') < total_nodes:
            # print(rospy.get_param('/nodes_ready')) # check how many nodes are ready
            rospy.sleep(0.1)  # Check every 100ms

        if rospy.is_shutdown(): break

        rospy.set_param('/trigger', False)
        rospy.set_param('/nodes_ready', 0)

if __name__ == '__main__':
    try:
        trigger()
    except rospy.ROSInterruptException:
        pass
