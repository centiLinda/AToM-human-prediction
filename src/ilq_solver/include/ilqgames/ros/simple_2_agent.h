#ifndef ILQGAMES_EXAMPLES_SIMPLE_2_AGENT_H
#define ILQGAMES_EXAMPLES_SIMPLE_2_AGENT_H

#include <ilqgames/solver/solver_params.h>
#include <ilqgames/solver/top_down_renderable_problem.h>
#include <vector>

namespace ilqgames {

struct Simple2AgentConfig{
  float x1_goal, y1_goal, x2_goal, y2_goal, v1_limit, v2_limit, 
        prox1_threshold, prox2_threshold, wGoal, wSpeed, wProximity, wControl;
};

class Simple2Agent : public TopDownRenderableProblem {
 public:
  ~Simple2Agent() {}
  Simple2Agent() : TopDownRenderableProblem() {}

  void SetupConfig(const Simple2AgentConfig& config){
    x1_goal = config.x1_goal;
    y1_goal = config.y1_goal;
    x2_goal = config.x2_goal;
    y2_goal = config.y2_goal;

    v1_limit = config.v1_limit;
    v2_limit = config.v2_limit;
    prox1_threshold = config.prox1_threshold;
    prox2_threshold = config.prox2_threshold;

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
};  // class Simple2Agent

}  // namespace ilqgames

#endif
