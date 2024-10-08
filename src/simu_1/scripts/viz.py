#!/usr/bin/env python3

import rospy
from helpers.msg import Point4D, Point4DArray, Point4DTwoArray
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import signal

position_h = []
position_r = []
position_h_pred = []
position_r_plan = []
h_pred_r = []

def human_position_callback(msg):
    global position_h
    position_h.append((msg.points[-1].x, msg.points[-1].y))

def robot_position_callback(msg):
    global position_r
    position_r.append((msg.x, msg.y))
    animate(0) # pred published before ukf, so mpc receive&plan before ukf

def predicted_human_position_callback(msg):
    global position_h_pred
    position_h_pred = [(p.x, p.y) for p in msg.points]

def robot_plan_callback(msg):
    global position_r_plan
    position_r_plan = [(p.x, p.y) for p in msg.points]

def h_pred_r_callback(msg):
    global h_pred_r
    h_pred_r = [(p.x, p.y) for p in msg.agent2_traj.points]

# Human-orange, Robot-blue, Prediction/Plan-dotted
def animate(i):
    fig, ax = plt.subplots(dpi=300)
    ax.set_xlim(-2, 13)
    ax.set_ylim(3, 11)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

    #------GT------
    if position_h:
        x_h, y_h = zip(*position_h) 
        plt.plot(x_h, y_h, marker='o', linestyle='-', color='orange', markersize=3, label='Human')

    if position_r:
        x_r, y_r = zip(*position_r) 
        plt.plot(x_r, y_r, marker='o', linestyle='-', color='royalblue', markersize=3, label='Robot')

    #------game solver prediction------
    if position_h_pred:
        x_h_p, y_h_p = zip(*position_h_pred) 
        plt.plot(x_h_p, y_h_p, marker='o', linestyle='--', color='orange', markersize=3, label='Human Prediction')

    if h_pred_r:
        x_h_pred_r, y_h_pred_r = zip(*h_pred_r)
        plt.plot(x_h_pred_r, y_h_pred_r, marker='o', linestyle='--', color='royalblue', markersize=3, label='Robot Prediction')

    # ------robot plan------
    if position_r_plan:
        x_r_p, y_r_p = zip(*position_r_plan) 
        plt.plot(x_r_p, y_r_p, marker='o', linestyle='--', color='green', markersize=3, label='Robot Plan')

    custom_legend = [Line2D([0], [0], color='orange', lw=2, label='Human'),
                     Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Human Prediction'),
                     Line2D([0], [0], color='royalblue', lw=2, label='Robot'),
                     Line2D([0], [0], color='green', lw=2, linestyle='--', label='Robot Plan'),
                     Line2D([0], [0], color='royalblue', lw=2, linestyle='--', label='Robot Prediction')]
    ax.legend(handles=custom_legend, loc='upper right')

    plt.savefig('PATH_TO_YOUR_WS/test.png')

def listener():
    rospy.init_node('viz')
    rospy.Subscriber('/current_robot', Point4D, robot_position_callback)
    rospy.Subscriber('/robot_plan', Point4DArray, robot_plan_callback)
    rospy.Subscriber('/observed_human', Point4DArray, human_position_callback)
    rospy.Subscriber('/predicted_human', Point4DArray, predicted_human_position_callback)
    rospy.Subscriber('/game_results', Point4DTwoArray, h_pred_r_callback)

def signal_handler(signal, frame):
    print('Exit plot')
    plt.close('all')

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    listener()
    rospy.spin()