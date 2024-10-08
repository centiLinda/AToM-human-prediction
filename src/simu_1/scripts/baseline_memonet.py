#!/usr/bin/env python3

import rospy
from helpers.msg import Point4D, Point4DArray
import numpy as np
import torch
from collections import deque
import random
from memonet.models.model_test_trajectory_res import model_encdec

''' 
Memonet (CVPR 2022) Baseline, adapted from https://github.com/MediaBrain-SJTU/MemoNet.git
'''

rand_seed = 0
np.random.seed(rand_seed)
random.seed(rand_seed)
torch.manual_seed(rand_seed)
torch.cuda.manual_seed_all(rand_seed)
device = torch.device('cpu')

# variables
prev_human = None
human_trajectory = deque(maxlen=3)
robot_trajectory = deque(maxlen=3)
human_updated = False
robot_updated = False
run_solver = False
goals = np.zeros(4) # x1, y1, x2, y2

# memonet only works well with pretrained 8-12 setting, we need to pad+resample to 3-10
settings = {"use_cuda": False,
            "dim_feature_tracklet": 8 * 2,
            "dim_feature_future": 12 * 2,
            "dim_embedding_key": int(24),
            "past_len": 8,
            "future_len": 12}

# Load memonet model
model_ae = torch.load('PATH_TO_YOUR_WS/src/simu_1/scripts/memonet/training/training_trajectory_original/model_encdec_trajectory', map_location=device)
mem_n2n = model_encdec(settings, model_ae)

# -----------------------------------Setup done-----------------------------------------------

def updateHumanCallback(human_Received):
    global human_updated, human_trajectory, prev_human

    # predict from previous, current is used for UKF update in AToM
    if len(human_Received.points) == 1:
        prev_human = human_Received.points[-1]
        human_trajectory.clear()
        return

    human_trajectory.append(prev_human)
    prev_human = human_Received.points[-1]
    human_updated = True

def updateRobotCallback(robot_Received):
    global robot_updated, robot_trajectory

    robot_trajectory.append(robot_Received)
    robot_updated = True

def predict_nn():
    global human_trajectory, robot_trajectory, goals

    prediction_horizon = 10

    while len(human_trajectory) < 3:
        human_trajectory.append(human_trajectory[-1])
    while len(robot_trajectory) < 3:
        robot_trajectory.append(robot_trajectory[-1])

    human_positions = np.array([[pt.x, pt.y] for pt in human_trajectory]) # [3, 2]
    robot_positions = np.array([[pt.x, pt.y] for pt in robot_trajectory])
    all_positions = np.stack([human_positions, robot_positions]) # [2, 3, 2]
    padding = np.repeat(all_positions[:, 0:1, :], 5, axis=1)
    all_positions = np.concatenate((padding, all_positions), axis=1) # [2, 8, 2]

    traj = torch.tensor(all_positions, dtype=torch.float32)
    goal = torch.tensor(np.array([goals[:2], goals[2:]]), dtype=torch.float32)
    
    initial_pos = traj[:, -1, :] # [n, 2]
    traj_relative = traj - traj[:, -1:, :] # [n, 8, 2]
    goal_relative = goal - initial_pos
    seq_start_end = [(0, 2)] # list of tuple
    
    # Inference
    relative_output = mem_n2n(traj_relative, traj, seq_start_end, initial_pos, goal_relative) # [2, 10, 2] [2, 12, 2]
    output = initial_pos.unsqueeze(1) + relative_output        

    human_predictions = output[0].detach().numpy() # (12, 2)
    original_indices = np.linspace(0, 12 - 1, 12) # [0, 1, ..., 11]
    new_indices = np.linspace(0, 12 - 1, 10) # [0, 1, ..., 9]
    human_predictions = np.array([np.interp(new_indices, original_indices, human_predictions[:, dim]) for dim in range(2)]).T

    current_pos = human_positions[-1:]
    human_predictions = np.vstack((current_pos, human_predictions))

    x = human_predictions[:, 0]
    y = human_predictions[:, 1]
    vx = np.append(np.diff(x), np.diff(x)[-1])
    vy = np.append(np.diff(y), np.diff(y)[-1])

    human_predictions_msg = Point4DArray()
    for i in range(prediction_horizon+1):
        point = Point4D()
        point.x = human_predictions[i][0]
        point.y = human_predictions[i][1]
        point.vx = vx[i]
        point.vy = vy[i]
        human_predictions_msg.points.append(point)

    return human_predictions_msg

def main():
    global run_solver, human_updated, robot_updated, goals
    
    rospy.init_node('prediction_memo', anonymous=True)
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
        
        if trigger and human_updated and robot_updated and not trigger_processed:
            run_solver = True
            human_updated = False
            robot_updated = False
            trigger_processed = True
        elif not trigger:
            trigger_processed = False
            run_solver = False
        else:
            run_solver = False
        
        if run_solver:
            human_prediction_msg = predict_nn() # (10, 2)
            pub_h.publish(human_prediction_msg)

            run_solver = False
            current_ready = rospy.get_param('/nodes_ready')
            rospy.set_param('/nodes_ready', current_ready + 1)
            
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
