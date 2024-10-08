#!/usr/bin/env python3

import rospy
from helpers.msg import Point4D, Point4DArray
import numpy as np
import math

def generate_trajectory(x_start, y_start, x_goal, y_goal, detour_strength, velocity, num_steps=11):
    # Linear interpolation for the base trajectory
    t = np.linspace(0, 1, num_steps)
    x_linear = (1 - t) * x_start + t * x_goal
    y_linear = (1 - t) * y_start + t * y_goal
    mid_point = (x_start + x_goal) / 2, (y_start + y_goal) / 2

    # Calculate perpendicular direction
    dx = x_goal - x_start
    dy = y_goal - y_start
    perp_dx, perp_dy = -dy, dx  # This rotates the direction vector by 90 degrees

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
    gaussian_detour = detour_strength * np.exp(-distances**2 / (2 * sigma**2))

    x_trajectory = x_linear + gaussian_detour * perp_dx
    y_trajectory = y_linear + gaussian_detour * perp_dy
    offset_x = x_trajectory[0] - x_start
    offset_y = y_trajectory[0] - y_start
    x_trajectory -= offset_x  # shift x axis back to starting point
    y_trajectory -= offset_y  # shift y axis back to starting point

    traj = [(x_start, y_start)]
    x_prev, y_prev = x_start, y_start
    for x, y in zip(x_trajectory, y_trajectory):
        while True:
            dx = x - x_prev
            dy = y - y_prev
            distance = math.sqrt(dx**2 + dy**2)
            if distance < velocity:
                break
            else:
                norm = math.sqrt(dx**2 + dy**2)
                dx, dy = (dx / norm) * velocity, (dy / norm) * velocity
                x_new, y_new = x_prev + dx, y_prev + dy
                traj.append((x_new, y_new))
                x_prev, y_prev = x_new, y_new            

    return traj

def talker():
    rospy.init_node('human')
    pub = rospy.Publisher('/observed_human', Point4DArray, queue_size=10, latch=True)
    rate = rospy.Rate(10) # check for trigger more frequently
    rospy.sleep(0.1)

    x_start = rospy.get_param('/x1_start')
    y_start = rospy.get_param('/y1_start')
    x_goal = rospy.get_param('/x1_goal')
    y_goal = rospy.get_param('/y1_goal')
    detour = rospy.get_param('/human_detour')
    velocity = rospy.get_param('/human_v')

    trajectory = generate_trajectory(x_start, y_start, x_goal, y_goal, detour, velocity)

    last = trajectory[-1]
    trajectory.extend([last] * 10) # let human wait at destination

    current_step = 0
    observed = Point4DArray()
    previous_pos = None
    trigger_processed = False

    # Initial position
    initial_position = trajectory[0]
    initial_msg = Point4D()
    initial_msg.header.stamp = rospy.Time.now()
    initial_msg.x, initial_msg.y = initial_position
    initial_msg.vx = trajectory[1][0] - trajectory[0][0]
    initial_msg.vy = trajectory[1][1] - trajectory[0][1]
    observed.points.append(initial_msg)
    pub.publish(observed)
    previous_pos = initial_position

    while not rospy.is_shutdown():
        trigger = rospy.get_param('/trigger', False)
        if trigger and not trigger_processed:
            if current_step < len(trajectory) - 1:
                current_step += 1
                position = trajectory[current_step]
                point_msg = Point4D()
                point_msg.header.stamp = rospy.Time.now()
                point_msg.x, point_msg.y = position
                point_msg.vx = position[0] - previous_pos[0]
                point_msg.vy = position[1] - previous_pos[1]

                observed.points.append(point_msg)
                if len(observed.points) > 10: observed.points.pop(0)
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
