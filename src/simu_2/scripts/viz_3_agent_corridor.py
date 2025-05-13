#!/usr/bin/env python3

import rospy
from helpers.msg import Point4D, Point4DArray, Point4DTwoArray, Point4DThreeArray
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import signal

position_h1 = []
position_h2 = []
position_h1_pred = []
position_h2_pred = []
position_h_pred_r = []
position_r = []
position_r_plan = []

width = 0.0
end_y = 0.0
right_x = 0.0

def human_position_callback(msg):
    global position_h1, position_h2
    position_h1 = [(p.x, p.y) for p in msg.agent1_traj.points]
    position_h2 = [(p.x, p.y) for p in msg.agent2_traj.points]

def h_pred_r_callback(msg):
    global position_h1_pred, position_h2_pred, position_h_pred_r
    position_h1_pred = [(p.x, p.y) for p in msg.agent1_traj.points]
    position_h2_pred = [(p.x, p.y) for p in msg.agent2_traj.points]
    position_h_pred_r = [(p.x, p.y) for p in msg.agent3_traj.points]

def robot_position_callback(msg):
    global position_r
    position_r.append((msg.x, msg.y))
    animate(0) # pred gets published before ukf, so mpc receive&plan before ukf

def robot_plan_callback(msg):
    global position_r_plan
    position_r_plan = [(p.x, p.y) for p in msg.points]

# rm static ending points to avoid tiny arrows
def filter_static_endings(x_h, y_h):
    filtered_x_h = [x_h[0]]
    filtered_y_h = [y_h[0]]
    for i in range(1, len(x_h)):
        if abs(x_h[i] - x_h[i-1]) > 0.2 or abs(y_h[i] - y_h[i-1]) > 0.2:
            filtered_x_h.append(x_h[i])
            filtered_y_h.append(y_h[i])
    if (len(filtered_x_h) < 2) or (len(filtered_y_h) < 2): # if too short
        filtered_x_h.append(x_h[-1])
        filtered_y_h.append(y_h[-1])

    return filtered_x_h, filtered_y_h

# wall is defined by (lower-left point, x-width, y-height)    
def draw_walls(ax):
    global width, end_y, right_x

    # left wall
    left_wall = patches.Rectangle((0.0, -width), width, 10.0, 
                                   linewidth=1, edgecolor='grey', facecolor='grey')
    # right wall
    right_wall = patches.Rectangle((right_x - width, -width), width, 10.0, 
                                   linewidth=1, edgecolor='grey', facecolor='grey')
    
    ax.add_patch(left_wall)
    ax.add_patch(right_wall)    

# Human1-orange, Human2-tomato, Robot-blue, Prediction/Plan-dotted
def animate(i):
    fig, ax = plt.subplots(figsize=(5, 6), dpi=300)
    ax.set_xlim(0, 7)
    ax.set_ylim(-1, 9)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.set_xticks(range(0, 5))  # From 0 to 4, inclusive
    ax.set_yticks(range(-1, 10))  # From 0 to 8, inclusive

    #------GT------
    if position_h1:
        x_h_1, y_h_1 = zip(*position_h1)
        ax.plot(x_h_1, y_h_1, marker='o', linestyle='-', color='orange', markersize=2, label='Human_1')

    if position_h2:
        x_h_2, y_h_2 = zip(*position_h2)
        ax.plot(x_h_2, y_h_2, marker='o', linestyle='-', color='tomato', markersize=2, label='Human_2')
    
    if position_r:
        x_r, y_r = zip(*position_r)
        ax.plot(x_r, y_r, marker='o', linestyle='-', color='royalblue', markersize=2, label='Robot')

    #------game solver prediction------
    if position_h1_pred:
        x_h_1, y_h_1 = zip(*position_h1_pred)
        ax.plot(x_h_1, y_h_1, marker='o', linestyle='--', color='orange', markersize=2, label='Human_1 Prediction')
        x_h_1, y_h_1 = filter_static_endings(x_h_1, y_h_1)
        ax.quiver(x_h_1[-2], y_h_1[-2], x_h_1[-1] - x_h_1[-2], y_h_1[-1] - y_h_1[-2],
                angles='xy', scale_units='xy', scale=1, color='orange', width=0.005, headlength=8, headwidth=8)
        
    if position_h2_pred:
        x_h_2, y_h_2 = zip(*position_h2_pred) 
        ax.plot(x_h_2, y_h_2, marker='o', linestyle='--', color='tomato', markersize=2, label='Human_2 Prediction')
        x_h_2, y_h_2 = filter_static_endings(x_h_2, y_h_2)
        ax.quiver(x_h_2[-2], y_h_2[-2], x_h_2[-1] - x_h_2[-2], y_h_2[-1] - y_h_2[-2],
                angles='xy', scale_units='xy', scale=1, color='tomato', width=0.005, headlength=8, headwidth=8)

    if position_h_pred_r:
        x_r, y_r = zip(*position_h_pred_r) 
        ax.plot(x_r, y_r, marker='o', linestyle='--', color='royalblue', markersize=2, label='Robot Prediction')
        x_r, y_r = filter_static_endings(x_r, y_r)
        ax.quiver(x_r[-2], y_r[-2], x_r[-1] - x_r[-2], y_r[-1] - y_r[-2],
                angles='xy', scale_units='xy', scale=1, color='royalblue', width=0.005, headlength=8, headwidth=8)

    # ------robot plan------
    if position_r_plan:
        x_r, y_r = zip(*position_r_plan) 
        ax.plot(x_r, y_r, marker='o', linestyle='--', color='green', markersize=2, label='Robot Plan')
    
    draw_walls(ax)

    custom_legend = [Line2D([0], [0], color='orange', lw=2, label='Human_1'),
                     Line2D([0], [0], color='tomato', lw=2, label='Human_2'),
                     Line2D([0], [0], color='royalblue', lw=2, label='Robot'),
                     Line2D([0], [0], color='black', lw=2, label='GT/Plan'),
                     Line2D([0], [0], color='black', lw=2, linestyle='--', label='Prediction'),
                     Line2D([0], [0], color='green', lw=2, linestyle='--', label='Robot Plan')]
    ax.legend(handles=custom_legend, loc='upper right')

    plt.savefig('PATH_TO_YOUR_WS/test.png')

def listener():
    global width, end_y, right_x

    rospy.init_node('viz')
    rospy.Subscriber('/game_results', Point4DThreeArray, h_pred_r_callback)
    rospy.Subscriber('/observed_human', Point4DTwoArray, human_position_callback)
    rospy.Subscriber('/current_robot', Point4D, robot_position_callback)
    rospy.Subscriber('/robot_plan', Point4DArray, robot_plan_callback)

    width = rospy.get_param('width')
    end_y = rospy.get_param('end_y')
    right_x = rospy.get_param('right_x')

def signal_handler(signal, frame):
    print('Exit plot')
    plt.close('all')

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    listener()
    rospy.spin()