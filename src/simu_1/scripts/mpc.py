#!/usr/bin/env python3

import rospy
from helpers.msg import Point4D, Point4DArray
import numpy as np
from threading import Thread

'''
Adapted from Pred2Nav (IROS 2023) https://github.com/sriyash421/Pred2Nav.git
'''

goal = np.zeros(2)
current_pos_global = Point4D()
latest_human_pred = None
trigger_processed = False
received_pred = False

def _wrap(angle):  # keep angle between [-pi, pi]
    while angle >= np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle
wrap = np.vectorize(_wrap)

def discrete_mpc(traj_h):
    curr_state = np.array([current_pos_global.x, current_pos_global.y])
    goal_state = goal
    stepCount = 0
    N = 10              # human prediction horizon
    control_sequence = Point4DArray()
    set_size = 50       # size of discrete trajectory set
    v_pref = 1.0        # preferred robot speed
    rollout_step = 6    # number of steps to calculate cost
    cost_weights = {
        'goal': 1.8,
        'human_avoid': 4.0
    }
    sigma_heading = 1.0
    sigma_side = 0.66
    sigma_rear = 0.33

    while stepCount <= (N - rollout_step): # planning horizon is 10-6+1=5
        # generate trajectory set (discrete MPC)
        angles = np.linspace(0, 2 * np.pi, set_size-1, endpoint=False)
        action_set = np.array([(np.cos(angle), np.sin(angle)) for angle in angles]) # unit vectors
        traj_set = np.array([curr_state + 
                            np.outer(np.arange(1, rollout_step+1), action * v_pref) 
                            for action in action_set]) # (set_size, rollout_step, 2)

        ##########
        # 1-Goal #
        ##########
        init_dist = np.linalg.norm(goal_state - curr_state)
        st_dist = np.clip(v_pref * np.arange(1, rollout_step+1), 0, init_dist)
        opt_dist = init_dist - st_dist # dist if follow straight path
        set_dist = np.linalg.norm(goal_state[None, None] - traj_set, axis=-1) # dist for all traj_set
        c_goal = np.sum((set_dist - opt_dist) / (2 * st_dist), axis=-1)
        c_goal = c_goal ** 2

        #####################
        # 2-Human avoidance #
        #####################
        c_human_avoid = 0
        if traj_h is not None:
            human_predict = []
            human_vel = []
            for i in range(stepCount+1, stepCount+1+rollout_step): # exclude current step
                human_pos = np.array([traj_h.points[i].x, traj_h.points[i].y])
                human_predict.append(human_pos)
                vel = np.array([traj_h.points[i].vx, traj_h.points[i].vy])
                human_vel.append(vel)
            human_predict = np.array(human_predict) # (6, 2)
            human_vel = np.array(human_vel)
            
            # distance from human at each step
            human_dist = traj_set - human_predict[None, :, :] # (10, 6, 2)
            # human headings
            obs_theta = np.arctan2(human_vel[:, 1], human_vel[:, 0]) # (6,)
            # check for static human
            static_obs = (np.linalg.norm(human_vel, axis=-1) < 0.01) # (6,) True/False
            # check if robot is in front of human
            relative_angle = np.arctan2(human_dist[:, :, 1], human_dist[:, :, 0]) - obs_theta + np.pi/2.0
            alpha = wrap(relative_angle) <= 0 # (10, 6) False if in front of human
            '''
            Asymmetric Gaussian social zone (Kirby 2010)
            sigma_heading = 1.0
            sigma_side = 0.66
            sigma_rear = 0.33
            '''
            sigma = np.where(alpha, sigma_rear, sigma_heading)
            sigma = static_obs + np.multiply(1-static_obs, sigma) # static human has no heading/rear
            sigma_s = 1.0 * static_obs + sigma_side * (1 - static_obs)
            obs_theta = obs_theta.reshape(1, -1) # (1, 6)
            sigma_s = sigma_s.reshape(1, -1)

            a = np.cos(obs_theta) ** 2 / (2 * sigma ** 2) + np.sin(obs_theta) ** 2 / (2 * sigma_s ** 2)
            b = np.sin(2 * obs_theta) / (4 * sigma ** 2) - np.sin(2 * obs_theta) / (4 * sigma_s ** 2)
            c = np.sin(obs_theta) ** 2 / (2 * sigma ** 2) + np.cos(obs_theta) ** 2 / (2 * sigma_s ** 2)
            c_human_avoid = np.exp(-((a * human_dist[:, :, 0] ** 2) 
                            + (2 * b * human_dist[:, :, 0] * human_dist[:, :, 1]) 
                            + (c * human_dist[:, :, 1] ** 2))) # (10, 6)
            c_human_avoid = np.sum(c_human_avoid, axis=-1)
            c_human_avoid = c_human_avoid ** 2

        #########
        # Total #
        #########
        c_total = cost_weights['goal'] * c_goal + cost_weights['human_avoid'] * c_human_avoid
        best_idx = np.argmin(c_total)
        best_traj = traj_set[best_idx][0]
        best_action = action_set[best_idx] * v_pref

        next_pos = Point4D()
        next_pos.header.stamp = rospy.Time.now() + rospy.Duration(stepCount)
        next_pos.x = best_traj[0]
        next_pos.y = best_traj[1]
        next_pos.vx = best_action[0]
        next_pos.vy = best_action[1]
        control_sequence.points.append(next_pos)

        curr_state = best_traj
        stepCount += 1

    return control_sequence

def planning():
    global latest_human_pred, trigger_processed, received_pred
    logged = False
    rate = rospy.Rate(10) # check for trigger more frequently
    while not rospy.is_shutdown():
        # Stop planning if reached goal already
        current_pos = np.array([current_pos_global.x, current_pos_global.y])
        if np.linalg.norm(goal - current_pos) <= 0.3 and not logged:
            rospy.loginfo("Goal reached!")
            logged = True
            break

        trigger = rospy.get_param('/trigger', False)
        if trigger and not trigger_processed and received_pred:
            traj_r = discrete_mpc(latest_human_pred) # latest_human_pred may be None
            pub.publish(traj_r)
            trigger_processed = True
            received_pred = False
            current_ready = rospy.get_param('/nodes_ready')
            rospy.set_param('/nodes_ready', current_ready + 1)                
        elif not trigger:
            trigger_processed = False

        rate.sleep()

def predicted_position_callback(msg):
    global latest_human_pred, received_pred
    latest_human_pred = msg
    received_pred = True

def update_current_pos(msg):
    global current_pos_global
    current_pos_global = msg

if __name__ == '__main__':
    try:
        rospy.init_node('mpc')
        rospy.Subscriber('/current_robot', Point4D, update_current_pos)
        pub = rospy.Publisher('/robot_plan', Point4DArray, queue_size=10)
        rospy.Subscriber('/predicted_human', Point4DArray, predicted_position_callback)

        goal[0] = rospy.get_param('/x2_goal')
        goal[1] = rospy.get_param('/y2_goal')

        planning_thread = Thread(target=planning)
        planning_thread.start()

        rospy.spin()
    except rospy.ROSInterruptException:
        pass
