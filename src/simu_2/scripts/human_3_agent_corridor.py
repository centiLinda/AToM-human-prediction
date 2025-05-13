#!/usr/bin/env python3

import rospy
from helpers.msg import Point4D, Point4DTwoArray
import numpy as np
import math

def generate_trajectory(states, num_steps=10):
    traj = [[], []]

    for i in range(2):
        idx = i*4
        # Linear interpolation for the base trajectory
        t = np.linspace(0, 1, num_steps)
        x_linear = (1 - t) * states[idx] + t * states[idx+2]
        y_linear = (1 - t) * states[idx+1] + t * states[idx+3]
        mid_point = (states[idx] + states[idx+2]) / 2, (states[idx+1] + states[idx+3]) / 2

        # Calculate perpendicular direction
        dx = states[idx+2] - states[idx]
        dy = states[idx+3] - states[idx+1]
        # +: 2 human curve towards right to avoid robot
        # -: 2 human curve towards left to avoid wall
        avoid_side = math.copysign(1, states[9 + i*2])
        perp_dx, perp_dy = (dy, -dx) if avoid_side > 0 else (-dy, dx) # (-, +) towards left, (+, -) towards right

        # Normalize the perpendicular vector
        length = math.sqrt(perp_dx**2 + perp_dy**2)
        if length == 0:
            # This handles the case where start and end points are the same
            perp_dx, perp_dy = 0, 0
        else:
            perp_dx, perp_dy = perp_dx / length, perp_dy / length

        # Gaussian function for detour along the perpendicular direction
        sigma = 5  # std deviation, controls shape of the overall distribution, don't change
        distances = np.sqrt((x_linear - mid_point[0])**2 + (y_linear - mid_point[1])**2)
        gaussian_detour = abs(states[9 + i*2]) * np.exp(-distances**2 / (2 * sigma**2))

        x_trajectory = x_linear + gaussian_detour * perp_dx
        y_trajectory = y_linear + gaussian_detour * perp_dy
        offset_x = x_trajectory[0] - states[idx]
        offset_y = y_trajectory[0] - states[idx+1]
        x_trajectory -= offset_x  # shift x axis back to starting point
        y_trajectory -= offset_y  # shift y axis back to starting point

        traj[i] = [(states[idx], states[idx+1])]
        x_prev, y_prev = states[idx], states[idx+1]
        for x, y in zip(x_trajectory, y_trajectory):
            while True:
                dx = x - x_prev
                dy = y - y_prev
                distance = math.sqrt(dx**2 + dy**2)
                if distance < states[8 + i*2]:
                    break
                else:
                    norm = math.sqrt(dx**2 + dy**2)
                    dx, dy = (dx / norm) * states[8 + i*2], (dy / norm) * states[8 + i*2]
                    x_new, y_new = x_prev + dx, y_prev + dy
                    traj[i].append((x_new, y_new))
                    x_prev, y_prev = x_new, y_new            

    return traj

def talker():
    rospy.init_node('human')
    pub = rospy.Publisher('/observed_human', Point4DTwoArray, queue_size=10, latch=True)
    rate = rospy.Rate(10) # check for trigger more frequently
    rospy.sleep(0.1)

    x1_start = rospy.get_param('/x1_start')
    y1_start = rospy.get_param('/y1_start')
    x1_goal = rospy.get_param('/x1_goal')
    y1_goal = rospy.get_param('/y1_goal')
    x2_start = rospy.get_param('/x2_start')
    y2_start = rospy.get_param('/y2_start')
    x2_goal = rospy.get_param('/x2_goal')
    y2_goal = rospy.get_param('/y2_goal')
    vel_1 = rospy.get_param('/vel_1')
    detour_1 = rospy.get_param('/detour_1')
    vel_2 = rospy.get_param('/vel_2')
    detour_2 = rospy.get_param('/detour_2')
    states = (x1_start, y1_start, x1_goal, y1_goal, 
             x2_start, y2_start, x2_goal, y2_goal, 
             vel_1, detour_1, vel_2, detour_2)

    trajectory = generate_trajectory(states) # list of 2 lists, with t tuples

    last = (trajectory[0][-1], trajectory[1][-1])
    trajectory[0].extend([last[0]] * 10) # let human wait at destination
    trajectory[1].extend([last[1]] * 10)

    current_step = 0
    observed = Point4DTwoArray()
    previous_pos = None
    trigger_processed = False

    # Initial position
    initial_position = (trajectory[0][0], trajectory[1][0])
    initial_msg_1 = Point4D()
    initial_msg_1.header.stamp = rospy.Time.now()
    initial_msg_1.x, initial_msg_1.y = initial_position[0]
    initial_msg_1.vx = trajectory[0][1][0] - trajectory[0][0][0]
    initial_msg_1.vy = trajectory[0][1][1] - trajectory[0][0][1]
    observed.agent1_traj.points.append(initial_msg_1)

    initial_msg_2 = Point4D()
    initial_msg_2.header.stamp = rospy.Time.now()
    initial_msg_2.x, initial_msg_2.y = initial_position[1]
    initial_msg_2.vx = trajectory[1][1][0] - trajectory[1][0][0]
    initial_msg_2.vy = trajectory[1][1][1] - trajectory[1][0][1]
    observed.agent2_traj.points.append(initial_msg_2)

    pub.publish(observed)
    previous_pos = initial_position

    while not rospy.is_shutdown():
        trigger = rospy.get_param('/trigger', False)
        if trigger and not trigger_processed:
            if current_step < len(trajectory[0]) - 1:
                current_step += 1
                position = (trajectory[0][current_step], trajectory[1][current_step])
                point_msg_1 = Point4D()
                point_msg_1.header.stamp = rospy.Time.now()
                point_msg_1.x, point_msg_1.y = position[0]
                point_msg_1.vx = position[0][0] - previous_pos[0][0]
                point_msg_1.vy = position[0][1] - previous_pos[0][1]
                observed.agent1_traj.points.append(point_msg_1)

                point_msg_2 = Point4D()
                point_msg_2.header.stamp = rospy.Time.now()
                point_msg_2.x, point_msg_2.y = position[1]
                point_msg_2.vx = position[1][0] - previous_pos[1][0]
                point_msg_2.vy = position[1][1] - previous_pos[1][1]
                observed.agent2_traj.points.append(point_msg_2)

                if len(observed.agent1_traj.points) > 10: 
                    observed.agent1_traj.points.pop(0)
                    observed.agent2_traj.points.pop(0)
                pub.publish(observed)
                previous_pos = position
                
                current_ready = rospy.get_param('/nodes_ready')
                rospy.set_param('/nodes_ready', current_ready + 1)
                trigger_processed = True

            else: 
                print('All human trajectories have been published')
        elif not trigger:
            trigger_processed = False
        
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
