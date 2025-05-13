#!/usr/bin/env python3

import os
import json
import rospy
from helpers.msg import Point4D, Point4DArray, Point4DTwoArray, Point4DThreeArray
import numpy as np
import pandas as pd
import torch
from collections import deque
from trajectron.model.model_registrar import ModelRegistrar
from trajectron.model.trajectron import Trajectron
from trajectron.environment import Environment, Scene, Node

''' 
Trajectron++ (ECCV 2020) Baseline, adapted from https://github.com/StanfordASL/Trajectron-plus-plus.git
'''

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# variables
prev_human_1, prev_human_2 = None, None
human_1_trajectory = deque(maxlen=3)
human_2_trajectory = deque(maxlen=3)
robot_trajectory = deque(maxlen=3)
human_updated = False
robot_updated = False
run_solver = False

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    }
}
data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
model_path = "PATH_TO_YOUR_WS/src/simu_2/scripts/trajectron/weights/eth_dyna_baseline_19_Aug"
checkpoint = 100

env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
attention_radius = dict()
attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
env.attention_radius = attention_radius
scene = Scene(timesteps=8+1, dt=1.0, name="eth_test", aug_func=None)# dummy scene
env.scenes = [scene]

model_registrar = ModelRegistrar(model_path, 'cpu')
model_registrar.load_models(checkpoint)
with open(os.path.join(model_path, 'config.json'), 'r') as config_json:
    hyperparams = json.load(config_json)

model = Trajectron(model_registrar, hyperparams, None, 'cpu')
model.set_environment(env)
model.set_annealing_params()

# -----------------------------------Setup done-----------------------------------------------

def get_all_values(d):
    for key, value in d.items():
        if isinstance(value, dict):
            yield from get_all_values(value)
        else:
            yield value

def updateHumanCallback(human_Received):
    global human_updated, human_1_trajectory, human_2_trajectory, prev_human_1, prev_human_2

    # predict from previous, current is used for UKF update in ToM
    if len(human_Received.agent1_traj.points) == 1:
        prev_human_1 = human_Received.agent1_traj.points[-1]
        prev_human_2 = human_Received.agent2_traj.points[-1]
        human_1_trajectory.clear()
        human_2_trajectory.clear()
        return

    human_1_trajectory.append(prev_human_1)
    human_2_trajectory.append(prev_human_2)
    prev_human_1 = human_Received.agent1_traj.points[-1]
    prev_human_2 = human_Received.agent2_traj.points[-1]
    human_updated = True

def updateRobotCallback(robot_Received):
    global robot_updated, robot_trajectory

    robot_trajectory.append(robot_Received)
    robot_updated = True

def predict_nn():
    global human_1_trajectory, human_2_trajectory, robot_trajectory

    prediction_horizon = 10

    while len(human_1_trajectory) < 3:
        human_1_trajectory.append(human_1_trajectory[-1])
        human_2_trajectory.append(human_2_trajectory[-1])
    while len(robot_trajectory) < 3:
        robot_trajectory.append(robot_trajectory[-1])

    scene = Scene(timesteps=3+1, dt=1.0, name="inference", aug_func=None)

    human_1_data_dict = {
        ('position', 'x'): np.array([pt.x for pt in human_1_trajectory]),
        ('position', 'y'): np.array([pt.y for pt in human_1_trajectory]),
        ('velocity', 'x'): np.array([pt.vx for pt in human_1_trajectory]),
        ('velocity', 'y'): np.array([pt.vy for pt in human_1_trajectory]),
        ('acceleration', 'x'): np.zeros(3),
        ('acceleration', 'y'): np.zeros(3)
    }
    human_1_data = pd.DataFrame(human_1_data_dict, columns=data_columns)
    human_1_node = Node(node_type=env.NodeType.PEDESTRIAN, node_id="0", data=human_1_data)
    human_1_node.first_timestep = 0
    scene.nodes.append(human_1_node)

    human_2_data_dict = {
        ('position', 'x'): np.array([pt.x for pt in human_2_trajectory]),
        ('position', 'y'): np.array([pt.y for pt in human_2_trajectory]),
        ('velocity', 'x'): np.array([pt.vx for pt in human_2_trajectory]),
        ('velocity', 'y'): np.array([pt.vy for pt in human_2_trajectory]),
        ('acceleration', 'x'): np.zeros(3),
        ('acceleration', 'y'): np.zeros(3)
    }
    human_2_data = pd.DataFrame(human_2_data_dict, columns=data_columns)
    human_2_node = Node(node_type=env.NodeType.PEDESTRIAN, node_id="1", data=human_2_data)
    human_2_node.first_timestep = 0
    scene.nodes.append(human_2_node)

    # add robot to scene
    robot_data_dict = {
        ('position', 'x'): np.array([pt.x for pt in robot_trajectory]),
        ('position', 'y'): np.array([pt.y for pt in robot_trajectory]),
        ('velocity', 'x'): np.array([pt.vx for pt in robot_trajectory]),
        ('velocity', 'y'): np.array([pt.vy for pt in robot_trajectory]),
        ('acceleration', 'x'): np.zeros(3),
        ('acceleration', 'y'): np.zeros(3)
    }
    robot_data = pd.DataFrame(robot_data_dict, columns=data_columns)
    robot_node = Node(node_type=env.NodeType.PEDESTRIAN, node_id="2", data=robot_data)
    robot_node.first_timestep = 0
    scene.nodes.append(robot_node)

    env.scenes = [scene]
    scene.calculate_scene_graph(env.attention_radius,
                                hyperparams['edge_addition_filter'],
                                hyperparams['edge_removal_filter'])

    # Inference
    timesteps = np.arange(scene.timesteps)
    predictions_dict = model.predict(scene,
                                     timesteps,
                                     ph=prediction_horizon,
                                     num_samples=1,
                                     min_history_timesteps=2,
                                     z_mode=False,
                                     gmm_mode=True,
                                     full_dist=False)
    predistions_list = list(get_all_values(predictions_dict))
    
    # Agent 1
    human_predictions = np.squeeze(predistions_list[0]) # (10, 2)
    current_pos = np.array([human_1_trajectory[-1].x, human_1_trajectory[-1].y])
    human_predictions = np.vstack((current_pos, human_predictions))
    x = human_predictions[:, 0]
    y = human_predictions[:, 1]
    vx = np.append(np.diff(x), np.diff(x)[-1])
    vy = np.append(np.diff(y), np.diff(y)[-1])
    human_predictions_msg_1 = Point4DArray()
    for i in range(prediction_horizon+1):
        point = Point4D()
        point.x = human_predictions[i][0]
        point.y = human_predictions[i][1]
        point.vx = vx[i]
        point.vy = vy[i]
        human_predictions_msg_1.points.append(point)

    # Agent 2
    human_predictions = np.squeeze(predistions_list[1]) # (10, 2)
    current_pos = np.array([human_2_trajectory[-1].x, human_2_trajectory[-1].y])
    human_predictions = np.vstack((current_pos, human_predictions))
    x = human_predictions[:, 0]
    y = human_predictions[:, 1]
    vx = np.append(np.diff(x), np.diff(x)[-1])
    vy = np.append(np.diff(y), np.diff(y)[-1])
    human_predictions_msg_2 = Point4DArray()
    for i in range(prediction_horizon+1):
        point = Point4D()
        point.x = human_predictions[i][0]
        point.y = human_predictions[i][1]
        point.vx = vx[i]
        point.vy = vy[i]
        human_predictions_msg_2.points.append(point)

    return human_predictions_msg_1, human_predictions_msg_2

def main():
    global run_solver, human_updated, robot_updated
    
    rospy.init_node('prediction_traj', anonymous=True)
    sub_h = rospy.Subscriber('/observed_human', Point4DTwoArray, updateHumanCallback)
    sub_r = rospy.Subscriber('/current_robot', Point4D, updateRobotCallback)
    pub_h = rospy.Publisher('/predicted_human', Point4DArray, queue_size=10)
    pub_all = rospy.Publisher('/game_results', Point4DThreeArray, queue_size=10)
    
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
            human_prediction_msg_1, human_prediction_msg_2 = predict_nn() # (10, 2)
            pub_h.publish(human_prediction_msg_1)

            # dummy robot prediction
            zero_trajectory = Point4DArray()
            for _ in range(len(human_prediction_msg_1.points)):
                zero_point = Point4D()
                zero_point.x = 0.0
                zero_point.y = 0.0
                zero_point.vx = 0.0
                zero_point.vy = 0.0
                zero_trajectory.points.append(zero_point)
            msg_total = Point4DThreeArray()
            msg_total.agent1_traj = human_prediction_msg_1
            msg_total.agent2_traj = human_prediction_msg_2
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
