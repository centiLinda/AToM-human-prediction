/*
Adapted from ILQGames (ICRA 2020) https://github.com/HJReachability/ilqgames.git
*/

#include <ilqgames/ros/simple_2_agent.h>
#include <ilqgames/solver/augmented_lagrangian_solver.h>
#include <ilqgames/solver/ilq_solver.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/utils/check_local_nash_equilibrium.h>
#include <ilqgames/utils/solver_log.h>
#include <ilqgames/utils/post_processing.h>

#include <stdio.h>

#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <helpers/Point4D.h>
#include <helpers/Point4DArray.h>
#include <helpers/Point4DTwoArray.h>
#include <helpers/SolverParams.h>
#include <vector>
#include <eigen3/Eigen/Dense>

#include <memory> // for std::shared_ptr
#include <kalman/UnscentedKalmanFilter.hpp>
#include <kalman/simple_2_agent_exchange.hpp>

ros::CallbackQueue robot_callback_queue; // for updating robot new position while in main loop

float x1_initial, y1_initial, vx1_initial, vy1_initial, x2_initial, y2_initial, vx2_initial, vy2_initial;
float x1_initial_previous, y1_initial_previous, vx1_initial_previous, vy1_initial_previous, 
      x2_initial_previous, y2_initial_previous, vx2_initial_previous, vy2_initial_previous;

bool human_updated = false;
bool robot_updated = false;
bool run_solver = false;

ilqgames::SolverParams params;
ilqgames::Simple2AgentConfig config;
std::shared_ptr<ilqgames::Simple2Agent> problem;

// UKF shortcuts
typedef ukfGame::State<float> State;
typedef ukfGame::ProcessModel<float> ProcessModel;
typedef ukfGame::Measurement<float> Measurement;
typedef ukfGame::MeasurementModel<float> MeasurementModel;

void updateRobotCallback(const helpers::Point4D& robot_Received){
  x2_initial_previous = x2_initial;
  y2_initial_previous = y2_initial;
  vx2_initial_previous = vx2_initial;
  vy2_initial_previous = vy2_initial;

  x2_initial = robot_Received.x;
  y2_initial = robot_Received.y;
  vx2_initial = robot_Received.vx;
  vy2_initial = robot_Received.vy;
  robot_updated = true;
}

void updateHumanCallback(const helpers::Point4DArray& human_Received){
  helpers::Point4D latest = human_Received.points.back();
  x1_initial = latest.x;
  y1_initial = latest.y;
  vx1_initial = latest.vx;
  vy1_initial = latest.vy;

  if (human_Received.points.size() > 1){
    helpers::Point4D second_latest = human_Received.points[human_Received.points.size() - 2];
    x1_initial_previous = second_latest.x;
    y1_initial_previous = second_latest.y;
    vx1_initial_previous = second_latest.vx;
    vy1_initial_previous = second_latest.vy;

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
  ros::Publisher pub_h = nh.advertise<helpers::Point4DArray>("/predicted_human", 10);
  ros::Publisher pub_all = nh.advertise<helpers::Point4DTwoArray>("/game_results", 10);
  ros::Publisher pub_param = nh.advertise<helpers::SolverParams>("/solver_params", 10);

  config.x1_goal = nh.param<float>("x1_goal", 0.0);
  config.y1_goal = nh.param<float>("y1_goal", 0.0);
  config.x2_goal = nh.param<float>("x2_goal", 0.0);
  config.y2_goal = nh.param<float>("y2_goal", 0.0);

  config.v1_limit = nh.param<float>("v1_limit", 0.0);
  config.v2_limit = nh.param<float>("v2_limit", 0.0);
  config.prox1_threshold = nh.param<float>("prox1_threshold", 0.0);
  config.prox2_threshold = nh.param<float>("prox2_threshold", 0.0);

  config.wGoal = nh.param<float>("wGoal", 0.0);
  config.wSpeed = nh.param<float>("wSpeed", 0.0);
  config.wProximity = nh.param<float>("wProximity", 0.0);
  config.wControl = nh.param<float>("wControl", 0.0);
  
  // Set up the game.
  problem = std::make_shared<ilqgames::Simple2Agent>();
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
                                       x2_initial, y2_initial, vx2_initial, vy2_initial}; // robot position not updated yet
      problem = std::make_shared<ilqgames::Simple2Agent>(); // need to reinitialize
      problem->SetupConfig(config); // update theta here
      problem->Initialize();
      problem->UpdateStates(new_states);
      ilqgames::AugmentedLagrangianSolver solver(problem, params);

      // solve the game
      std::shared_ptr<const ilqgames::SolverLog> log = solver.Solve();
      size_t iteration = log->NumIterates();
      Eigen::MatrixXf results = log->GetPrediction(iteration-1); // 10*8

      // -----------------------------Post-processing-----------------------------------
      ilqgames::TrajectoryProcessor processor;
      Eigen::MatrixXf processed_trajectory = processor.processTrajectory(results, config.v1_limit, config.v2_limit);
      results = processed_trajectory;

      // publish
      helpers::Point4DArray msg_h, msg_r;
      for (int i=0; i < results.rows(); ++i){ // include current position
          helpers::Point4D point_h, point_r;
          point_h.x = results(i, 0);
          point_h.y = results(i, 1);
          point_h.vx = results(i, 2);
          point_h.vy = results(i, 3);
          msg_h.points.push_back(point_h);

          point_r.x = results(i, 4);
          point_r.y = results(i, 5);
          point_r.vx = results(i, 6);
          point_r.vy = results(i, 7);
          msg_r.points.push_back(point_r);
      }
      pub_h.publish(msg_h);

      helpers::Point4DTwoArray msg_total;
      msg_total.agent1_traj = msg_h;
      msg_total.agent2_traj = msg_r;
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
                                            vx2_initial_previous, vy2_initial_previous};
      problem->Initialize();
      problem->UpdateStates(previous_states);

      State x;
      x << config.prox1_threshold, config.prox2_threshold, config.v1_limit, config.v2_limit;
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
      x_pred = ukf.update(measure, measurement);
      cout << "--------------prox1: " << x_pred.prox1_threshold() << endl;
      cout << "--------------prox2: " << x_pred.prox2_threshold() << endl;
      cout << "--------------v1: " << x_pred.v1_limit() << endl;
      cout << "--------------v2: " << x_pred.v2_limit() << endl;
      config.prox1_threshold = x_pred.prox1_threshold();
      config.prox2_threshold = x_pred.prox2_threshold();
      config.v1_limit = x_pred.v1_limit();
      config.v2_limit = x_pred.v2_limit();

      helpers::SolverParams msg_param;
      msg_param.prox1 = x_pred.prox1_threshold();
      msg_param.prox2 = x_pred.prox2_threshold();
      msg_param.v1 = x_pred.v1_limit();
      msg_param.v2 = x_pred.v2_limit();
      pub_param.publish(msg_param);
      rate.sleep(); // trigger sometimes stuck

      nh.getParam("/nodes_ready", current_ready);
      nh.setParam("/nodes_ready", current_ready + 1);
    }

    rate.sleep();
    ros::spinOnce();
  }

}
