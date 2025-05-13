/*
Adapted from ILQGames (ICRA 2020) https://github.com/HJReachability/ilqgames.git
*/

#include <ilqgames/ros/corridor_3_agent.h>
#include <ilqgames/solver/augmented_lagrangian_solver.h>
#include <ilqgames/solver/ilq_solver.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/utils/check_local_nash_equilibrium.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/post_processing_3_agent.h>

#include <stdio.h>

#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <helpers/Point4D.h>
#include <helpers/Point4DArray.h>
#include <helpers/Point4DTwoArray.h>
#include <helpers/Point4DThreeArray.h>
#include <helpers/SolverParams3Agent.h>
#include <vector>
#include <eigen3/Eigen/Dense>

#include <memory> // for std::shared_ptr
#include <kalman/UnscentedKalmanFilter.hpp>
#include <kalman/corridor_3_agent.hpp>

ros::CallbackQueue robot_callback_queue; // for updating robot new position while in main loop

float x1_initial, y1_initial, vx1_initial, vy1_initial, x2_initial, y2_initial, vx2_initial, vy2_initial,
      x3_initial, y3_initial, vx3_initial, vy3_initial;
float x1_initial_previous, y1_initial_previous, vx1_initial_previous, vy1_initial_previous, 
      x2_initial_previous, y2_initial_previous, vx2_initial_previous, vy2_initial_previous,
      x3_initial_previous, y3_initial_previous, vx3_initial_previous, vy3_initial_previous;

bool human_updated = false;
bool robot_updated = false;
bool run_solver = false;

ilqgames::SolverParams params;
ilqgames::Corridor3AgentConfig config;
std::shared_ptr<ilqgames::Corridor3Agent> problem;

// UKF shortcuts
typedef ukfGame::State<float> State;
typedef ukfGame::ProcessModel<float> ProcessModel;
typedef ukfGame::Measurement<float> Measurement;
typedef ukfGame::MeasurementModel<float> MeasurementModel;

void updateRobotCallback(const helpers::Point4D& robot_Received){
  x3_initial_previous = x3_initial;
  y3_initial_previous = y3_initial;
  vx3_initial_previous = vx3_initial;
  vy3_initial_previous = vy3_initial;

  x3_initial = robot_Received.x;
  y3_initial = robot_Received.y;
  vx3_initial = robot_Received.vx;
  vy3_initial = robot_Received.vy;
  robot_updated = true;
}

void updateHumanCallback(const helpers::Point4DTwoArray& human_Received){
  helpers::Point4D latest_1 = human_Received.agent1_traj.points.back();
  x1_initial = latest_1.x;
  y1_initial = latest_1.y;
  vx1_initial = latest_1.vx;
  vy1_initial = latest_1.vy;
  helpers::Point4D latest_2 = human_Received.agent2_traj.points.back();
  x2_initial = latest_2.x;
  y2_initial = latest_2.y;
  vx2_initial = latest_2.vx;
  vy2_initial = latest_2.vy;

  if (human_Received.agent1_traj.points.size() > 1){
    helpers::Point4D second_latest_1 = human_Received.agent1_traj.points[human_Received.agent1_traj.points.size() - 2];
    x1_initial_previous = second_latest_1.x;
    y1_initial_previous = second_latest_1.y;
    vx1_initial_previous = second_latest_1.vx;
    vy1_initial_previous = second_latest_1.vy;
    helpers::Point4D second_latest_2 = human_Received.agent2_traj.points[human_Received.agent2_traj.points.size() - 2];
    x2_initial_previous = second_latest_2.x;
    y2_initial_previous = second_latest_2.y;
    vx2_initial_previous = second_latest_2.vx;
    vy2_initial_previous = second_latest_2.vy;

    human_updated = true;
  }
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "prediction_atom");
  ros::NodeHandle nh;
  ros::NodeHandle robot_nh;
  robot_nh.setCallbackQueue(&robot_callback_queue);

  ros::Subscriber sub_h = nh.subscribe("/observed_human", 10, updateHumanCallback);
  ros::Subscriber sub_r = robot_nh.subscribe("/current_robot", 10, updateRobotCallback);
  ros::Publisher pub_h = nh.advertise<helpers::Point4DArray>("/predicted_human", 10); // for logging purpose
  ros::Publisher pub_all = nh.advertise<helpers::Point4DThreeArray>("/game_results", 10);
  ros::Publisher pub_param = nh.advertise<helpers::SolverParams3Agent>("/solver_params", 10);

  config.x1_goal = nh.param<float>("x1_goal", 0.0);
  config.y1_goal = nh.param<float>("y1_goal", 0.0);
  config.x2_goal = nh.param<float>("x2_goal", 0.0);
  config.y2_goal = nh.param<float>("y2_goal", 0.0);
  config.x3_goal = nh.param<float>("x3_goal", 0.0);
  config.y3_goal = nh.param<float>("y3_goal", 0.0);

  config.v1_limit = nh.param<float>("v1_limit", 0.0);
  config.v2_limit = nh.param<float>("v2_limit", 0.0);
  config.v3_limit = nh.param<float>("v3_limit", 0.0);
  config.prox1_threshold = nh.param<float>("prox1_threshold", 0.0);
  config.prox2_threshold = nh.param<float>("prox2_threshold", 0.0);
  config.prox3_threshold = nh.param<float>("prox3_threshold", 0.0);

  config.wGoal = nh.param<float>("wGoal", 0.0);
  config.wSpeed = nh.param<float>("wSpeed", 0.0);
  config.wProximity = nh.param<float>("wProximity", 0.0);
  config.wControl = nh.param<float>("wControl", 0.0);
  
  // Set up the game.
  problem = std::make_shared<ilqgames::Corridor3Agent>();
  problem->SetupConfig(config);
  problem->Initialize();

  bool trigger = false;
  bool trigger_processed = false;
  int current_ready;

  ros::Rate rate(10); // check for trigger more frequently
  while (ros::ok()){
    ros::getGlobalCallbackQueue()->callAvailable(ros::WallDuration(0.01));
    robot_callback_queue.callAvailable(ros::WallDuration(0.01));

    nh.getParam("/trigger", trigger);
    if (trigger && human_updated && robot_updated && !trigger_processed) {
      run_solver = true;
      human_updated = false;
      robot_updated = false;
      trigger_processed = true;
    }
    else if (!trigger){
      trigger_processed = false;
      run_solver = false;
    }
    else {
      run_solver = false;
    }

    if (run_solver){       
      // update latest states
      std::vector<float> new_states = {x1_initial_previous, y1_initial_previous, 
                                       vx1_initial_previous, vy1_initial_previous,
                                       x2_initial_previous, y2_initial_previous, 
                                       vx2_initial_previous, vy2_initial_previous,
                                       x3_initial, y3_initial, vx3_initial, vy3_initial}; // robot position not updated yet
      problem = std::make_shared<ilqgames::Corridor3Agent>(); // need to reinitialize
      problem->SetupConfig(config); // update theta here
      problem->Initialize();
      problem->UpdateStates(new_states);
      ilqgames::AugmentedLagrangianSolver solver(problem, params);

      // solve the game
      std::shared_ptr<const ilqgames::SolverLog> log = solver.Solve();
      size_t iteration = log->NumIterates();
      Eigen::MatrixXf results = log->GetPrediction(iteration-1); // 10*8

      // -----------------------------Post-processing-----------------------------------
      Eigen::Vector2f goal_1(config.x1_goal, config.y1_goal);
      Eigen::Vector2f goal_2(config.x2_goal, config.y2_goal);
      Eigen::Vector2f goal_3(config.x3_goal, config.y3_goal);
      ilqgames::TrajectoryProcessor processor;
      Eigen::MatrixXf processed_trajectory = processor.processTrajectory(results, goal_1, goal_2, goal_3, 
                                                                         config.v1_limit, config.v2_limit, config.v3_limit);
      results = processed_trajectory;

      // publish
      helpers::Point4DArray msg_h1, msg_h2, msg_r;
      for (int i=0; i < results.rows(); ++i){ // include current position
          helpers::Point4D point_h1, point_h2, point_r;
          point_h1.x = results(i, 0);
          point_h1.y = results(i, 1);
          point_h1.vx = results(i, 2);
          point_h1.vy = results(i, 3);
          msg_h1.points.push_back(point_h1);

          point_h2.x = results(i, 4);
          point_h2.y = results(i, 5);
          point_h2.vx = results(i, 6);
          point_h2.vy = results(i, 7);
          msg_h2.points.push_back(point_h2);

          point_r.x = results(i, 8);
          point_r.y = results(i, 9);
          point_r.vx = results(i, 10);
          point_r.vy = results(i, 11);
          msg_r.points.push_back(point_r);
      }
      pub_h.publish(msg_h1);

      helpers::Point4DThreeArray msg_total;
      msg_total.agent1_traj = msg_h1;
      msg_total.agent2_traj = msg_h2;
      msg_total.agent3_traj = msg_r;
      pub_all.publish(msg_total);

      // wait for mpc to send back 
      while(!robot_updated){
        robot_callback_queue.callAvailable(ros::WallDuration(0.1));
        rate.sleep();
      }

      // -----------------------------UKF-----------------------------------
      cout << "UKF:" << endl;
      std::vector<float> previous_states = {x1_initial_previous, y1_initial_previous,
                                            vx1_initial_previous, vy1_initial_previous,
                                            x2_initial_previous, y2_initial_previous,
                                            vx2_initial_previous, vy2_initial_previous,
                                            x3_initial_previous, y3_initial_previous,
                                            vx3_initial_previous, vy3_initial_previous};
      problem->Initialize();
      problem->UpdateStates(previous_states);

      State x;
      x << config.prox1_threshold, config.prox2_threshold, config.prox3_threshold,
           config.v1_limit, config.v2_limit, config.v3_limit;
      ProcessModel sys;
      MeasurementModel measure(params, config, problem);
      Kalman::UnscentedKalmanFilter<State> ukf(1.0); // alpha=0.1, beta=2, kappa=0
      ukf.init(x);
      auto x_pred = ukf.predict(sys);

      // UKF correct
      Measurement measurement;
      measurement.x1() = x1_initial;
      measurement.y1() = y1_initial;
      measurement.vx1() = vx1_initial;
      measurement.vy1() = vy1_initial;
      measurement.x2() = x2_initial;
      measurement.y2() = y2_initial;
      measurement.vx2() = vx2_initial;
      measurement.vy2() = vy2_initial;
      measurement.x3() = x3_initial;
      measurement.y3() = y3_initial;
      measurement.vx3() = vx3_initial;
      measurement.vy3() = vy3_initial;
      x_pred = ukf.update(measure, measurement);
      cout << "--------------prox1: " << x_pred.prox1_threshold() << endl;
      cout << "--------------prox2: " << x_pred.prox2_threshold() << endl;
      cout << "--------------prox3: " << x_pred.prox3_threshold() << endl;
      cout << "--------------v1: " << x_pred.v1_limit() << endl;
      cout << "--------------v2: " << x_pred.v2_limit() << endl;
      cout << "--------------v3: " << x_pred.v3_limit() << endl;
      config.prox1_threshold = x_pred.prox1_threshold();
      config.prox2_threshold = x_pred.prox2_threshold();
      config.prox3_threshold = x_pred.prox3_threshold();
      config.v1_limit = x_pred.v1_limit();
      config.v2_limit = x_pred.v2_limit();
      config.v3_limit = x_pred.v3_limit();

      helpers::SolverParams3Agent msg_param;
      msg_param.prox1 = x_pred.prox1_threshold();
      msg_param.prox2 = x_pred.prox2_threshold();
      msg_param.prox3 = x_pred.prox3_threshold();
      msg_param.v1 = x_pred.v1_limit();
      msg_param.v2 = x_pred.v2_limit();
      msg_param.v3 = x_pred.v3_limit();
      pub_param.publish(msg_param);
      rate.sleep(); // trigger sometimes stuck

      nh.getParam("/nodes_ready", current_ready);
      nh.setParam("/nodes_ready", current_ready + 1);
    }

    rate.sleep();
    ros::spinOnce();
  }

}
