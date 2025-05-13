#ifndef ILQGAMES_EXAMPLES_CORRIDOR_3_AGENT_H
#define ILQGAMES_EXAMPLES_CORRIDOR_3_AGENT_H

#include <ilqgames/solver/solver_params.h>
#include <ilqgames/solver/top_down_renderable_problem.h>
#include <vector>

namespace ilqgames {

struct Corridor3AgentConfig{
  float x1_goal, y1_goal, x2_goal, y2_goal, v1_limit, v2_limit, 
        prox1_threshold, prox2_threshold, wGoal, wSpeed, wProximity, wControl,
        x3_goal, y3_goal, v3_limit, prox3_threshold,
        width, end_y, right_x, wObstacle;
};

class Corridor3Agent : public TopDownRenderableProblem {
 public:
  ~Corridor3Agent() {}
  Corridor3Agent() : TopDownRenderableProblem() {}

  void SetupConfig(const Corridor3AgentConfig& config){
    x1_goal = config.x1_goal;
    y1_goal = config.y1_goal;
    x2_goal = config.x2_goal;
    y2_goal = config.y2_goal;
    x3_goal = config.x3_goal;
    y3_goal = config.y3_goal;

    v1_limit = config.v1_limit;
    v2_limit = config.v2_limit;
    v3_limit = config.v3_limit;
    prox1_threshold = config.prox1_threshold;
    prox2_threshold = config.prox2_threshold;
    prox3_threshold = config.prox3_threshold;

    width = config.width;
    end_y = config.end_y;
    right_x = config.right_x;
    wObstacle = config.wObstacle;

    wGoal = config.wGoal;
    wSpeed = config.wSpeed;
    wProximity = config.wProximity;
    wControl = config.wControl;

    ConstructDynamics();
    ConstructPlayerCosts();
  }

  void UpdateStates(const std::vector<float>& states){
    x1_initial = states[0];
    y1_initial = states[1];
    vx1_initial = states[2];
    vy1_initial = states[3];
    x2_initial = states[4];
    y2_initial = states[5];
    vx2_initial = states[6];
    vy2_initial = states[7];
    x3_initial = states[8];
    y3_initial = states[9];
    vx3_initial = states[10];
    vy3_initial = states[11];

    ConstructInitialState();
  }

  // Construct dynamics, initial state, and player costs.
  void ConstructDynamics();
  void ConstructInitialState();
  void ConstructPlayerCosts();

  // Unpack x, y, heading (for each player, potentially) from a given state.
  std::vector<float> Xs(const VectorXf& x) const;
  std::vector<float> Ys(const VectorXf& x) const;
  std::vector<float> Thetas(const VectorXf& x) const;

 private:
  float x1_goal, y1_goal, x2_goal, y2_goal, v1_limit, v2_limit, 
        prox1_threshold, prox2_threshold, wGoal, wSpeed, wProximity, wControl;
  float x1_initial, y1_initial, vx1_initial, vy1_initial, 
        x2_initial, y2_initial, vx2_initial, vy2_initial;
  float x3_goal, y3_goal, v3_limit, prox3_threshold, x3_initial, y3_initial, vx3_initial, vy3_initial;
  float width, end_y, right_x, wObstacle;

};  // class Corridor3Agent

}  // namespace ilqgames

#endif
