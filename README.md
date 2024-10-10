# AToM: Adaptive Theory-of-Mind-Based Human Motion Prediction in Long-Term Human-Robot Interactions

**NOTE**: More content to be added after paper acceptance. Feel free to post an issue if you have any questions.

## Getting started

This code has been tested on [ROS1](https://wiki.ros.org/ROS/Tutorials) Noetic with Python 3.8. 
It contains 5 ROS packages:
* `helpers`: Defines customised messages and helper scripts.
* `ilq_solver`: Modified from [ILQGames](https://github.com/HJReachability/ilqgames.git). It is our human internal model that generates predicted human trajectories and human-predicted robot trajectories. We use [this](https://github.com/mherb/kalman.git) UKF implementation. There is a separate executable for each experiment scenario.
* `simu_1`, `simu_2`, `simu_3`: Contains launch files, baseline prediction methods([SocialForce](https://github.com/yuxiang-gao/PySocialForce.git), [Memonet](https://github.com/MediaBrain-SJTU/MemoNet.git), [Trajectron++](https://github.com/StanfordASL/Trajectron-plus-plus.git)), [MPC](https://github.com/sriyash421/Pred2Nav.git) planner, and other experiment components. 
For Memonet and Trajectron++, download the weights from the respective repos and place them under `memonet/training/` and `trajectron/weights/`. Replace all the `PATH_TO_YOUR_WS` in the code with your local path.
To build the packages, use [catkin](https://wiki.ros.org/catkin/Tutorials) or [catkin_tools](https://catkin-tools.readthedocs.io/en/latest/verbs/catkin_build.html).

## Experiments
To run an experiment (scenario 1, for example):
```
roslaunch simu_1 2_agent_exchange.launch
```
The visualisation image is saved as `your_ws/test.png` by default. We choose a step-by-step mechanism so it's easy to investigate the result at each step. You will be prompted to press 'Enter' to execute the next step. You may remove this mechanism to let it run normally.
To repeat the experiment:
```
./restart_simu_1.sh -d xx -v xx
```
where `-d` and `-v` modify the detour and speed of the simulated human. The values used in the paper can be found in each config file in `ilq_solver\config\`.

## TODO
- [] links to paper & supplementary video
- [] acknowledgement & citation
- [] simu_2 & simu_3 pkg