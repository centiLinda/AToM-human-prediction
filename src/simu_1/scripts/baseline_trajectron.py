#!/usr/bin/env python3

import sys
import os
import json
import rospy
from helpers.msg import Point4D, Point4DArray
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
prev_human = None
human_trajectory = deque(maxlen=3)
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
model_path = "PATH_TO_YOUR_WS/src/simu_1/scripts/trajectron/weights/eth_dyna_baseline_19_Aug"
checkpoint = 100

env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
attention_radius = dict()
attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
env.attention_radius = attention_radius
scene = Scene(timesteps=8+1, dt=1.0, name="eth_test", aug_func=None) # dummy scene
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
    global human_trajectory, robot_trajectory

    prediction_horizon = 10

    while len(human_trajectory) < 3:
        human_trajectory.append(human_trajectory[-1])
    while len(robot_trajectory) < 3:
        robot_trajectory.append(robot_trajectory[-1])

    scene = Scene(timesteps=3+1, dt=1.0, name="inference", aug_func=None)
    human_data_dict = {
        ('position', 'x'): np.array([pt.x for pt in human_trajectory]),
        ('position', 'y'): np.array([pt.y for pt in human_trajectory]),
        ('velocity', 'x'): np.array([pt.vx for pt in human_trajectory]),
        ('velocity', 'y'): np.array([pt.vy for pt in human_trajectory]),
        ('acceleration', 'x'): np.zeros(3),
        ('acceleration', 'y'): np.zeros(3)
    }
    human_data = pd.DataFrame(human_data_dict, columns=data_columns)
    human_node = Node(node_type=env.NodeType.PEDESTRIAN, node_id="0", data=human_data)
    human_node.first_timestep = 0
    scene.nodes.append(human_node)

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
    robot_node = Node(node_type=env.NodeType.PEDESTRIAN, node_id="1", data=robot_data)
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
    human_predictions = np.squeeze(predistions_list[0]) # (10, 2)

    current_pos = np.array([human_trajectory[-1].x, human_trajectory[-1].y])
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
    global run_solver, human_updated, robot_updated
    
    rospy.init_node('prediction_traj', anonymous=True)
    sub_h = rospy.Subscriber('/observed_human', Point4DArray, updateHumanCallback)
    sub_r = rospy.Subscriber('/current_robot', Point4D, updateRobotCallback)
    pub_h = rospy.Publisher('/predicted_human', Point4DArray, queue_size=10)
    
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
