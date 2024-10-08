#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/cost/semiquadratic_norm_cost.h>
#include <ilqgames/cost/proximity_cost.h>
#include <ilqgames/cost/final_time_cost.h>
#include <ilqgames/cost/time_weighted_cost.h>
#include <ilqgames/constraint/single_dimension_constraint.h>

#include <ilqgames/dynamics/concatenated_dynamical_system.h>
#include <ilqgames/dynamics/single_player_point_mass_2d.h>

#include <ilqgames/ros/simple_2_agent.h>

#include <ilqgames/solver/ilq_solver.h>
#include <ilqgames/solver/problem.h>
#include <ilqgames/solver/solver_params.h>
#include <ilqgames/utils/types.h>

#include <math.h>
#include <memory>
#include <vector>

#include <ros/ros.h>

namespace ilqgames {
namespace {
// State dimensions.
using Dyn = SinglePlayerPointMass2D;
static const Dimension kP1PxIdx = Dyn::kPxIdx;
static const Dimension kP1PyIdx = Dyn::kPyIdx;
static const Dimension kP1VxIdx = Dyn::kVxIdx;
static const Dimension kP1VyIdx = Dyn::kVyIdx;

static const Dimension kP2PxIdx = Dyn::kNumXDims + Dyn::kPxIdx;
static const Dimension kP2PyIdx = Dyn::kNumXDims + Dyn::kPyIdx;
static const Dimension kP2VxIdx = Dyn::kNumXDims + Dyn::kVxIdx;
static const Dimension kP2VyIdx = Dyn::kNumXDims + Dyn::kVyIdx;
}  // anonymous namespace

void Simple2Agent::ConstructDynamics() {
  dynamics_.reset(new ConcatenatedDynamicalSystem(
      {std::make_shared<Dyn>(), 
       std::make_shared<Dyn>()}));
}

void Simple2Agent::ConstructInitialState() {
  x0_ = VectorXf::Zero(dynamics_->XDim());
  x0_(kP1PxIdx) = this->x1_initial;
  x0_(kP1PyIdx) = this->y1_initial;
  x0_(kP1VxIdx) = this->vx1_initial;
  x0_(kP1VyIdx) = this->vy1_initial;

  x0_(kP2PxIdx) = this->x2_initial;
  x0_(kP2PyIdx) = this->y2_initial;
  x0_(kP2VxIdx) = this->vx2_initial;
  x0_(kP2VyIdx) = this->vy2_initial;
}

void Simple2Agent::ConstructPlayerCosts() {
  // Set up costs for all players.
  player_costs_.emplace_back("P1", 1.0, 0.0);
  player_costs_.emplace_back("P2", 1.0, 0.0);
  auto& p1_cost = player_costs_[0];
  auto& p2_cost = player_costs_[1];

  // 1. goal cost
  constexpr float FinalTimeWindow = 11.0; // Time-weighted cost (later timesteps get higher weights)
  const auto p1_goalx_cost = std::make_shared<TimeWeightedCost>(
    std::make_shared<QuadraticCost>(this->wGoal, kP1PxIdx, this->x1_goal), 
    time::kTimeHorizon - FinalTimeWindow, "GoalX1");
  const auto p1_goaly_cost = std::make_shared<TimeWeightedCost>(
    std::make_shared<QuadraticCost>(this->wGoal, kP1PyIdx, this->y1_goal), 
    time::kTimeHorizon - FinalTimeWindow, "GoalY1");
  p1_cost.AddStateCost(p1_goalx_cost);
  p1_cost.AddStateCost(p1_goaly_cost);

  const auto p2_goalx_cost = std::make_shared<TimeWeightedCost>(
    std::make_shared<QuadraticCost>(this->wGoal, kP2PxIdx, this->x2_goal), 
    time::kTimeHorizon - FinalTimeWindow, "GoalX2");
  const auto p2_goaly_cost = std::make_shared<TimeWeightedCost>(
    std::make_shared<QuadraticCost>(this->wGoal, kP2PyIdx, this->y2_goal), 
    time::kTimeHorizon - FinalTimeWindow, "GoalY2");
  p2_cost.AddStateCost(p2_goalx_cost);
  p2_cost.AddStateCost(p2_goaly_cost);

  // 2. speed cost
  const std::shared_ptr<SemiquadraticNormCost> p1_v_cost(
    new SemiquadraticNormCost(this->wSpeed, {kP1VxIdx, kP1VyIdx}, this->v1_limit, true, "Max_Speed1"));
  const std::shared_ptr<SemiquadraticNormCost> p2_v_cost(
    new SemiquadraticNormCost(this->wSpeed, {kP2VxIdx, kP2VyIdx}, this->v2_limit, true, "Max_Speed2"));
  p1_cost.AddStateCost(p1_v_cost);
  p2_cost.AddStateCost(p2_v_cost);

  // 3. proximity cost
  const std::shared_ptr<ProximityCost> p1_proximity_cost(
    new ProximityCost(this->wProximity, {kP1PxIdx, kP1PyIdx}, {kP2PxIdx, kP2PyIdx}, this->prox1_threshold, "ProximityP1"));
  const std::shared_ptr<ProximityCost> p2_proximity_cost(
    new ProximityCost(this->wProximity, {kP2PxIdx, kP2PyIdx}, {kP1PxIdx, kP1PyIdx}, this->prox2_threshold, "ProximityP2"));
  p1_cost.AddStateCost(p1_proximity_cost);
  p2_cost.AddStateCost(p2_proximity_cost);

  // 4. control cost
  const auto control_cost = std::make_shared<QuadraticCost>(this->wControl, -1, 0.0, "ControlCost");
  p1_cost.AddControlCost(0, control_cost);
  p2_cost.AddControlCost(1, control_cost);
}

inline std::vector<float> Simple2Agent::Xs(const VectorXf& x) const {
  return {x(kP1PxIdx), x(kP2PxIdx)};
}

inline std::vector<float> Simple2Agent::Ys(const VectorXf& x) const {
  return {x(kP1PyIdx), x(kP2PyIdx)};
}

inline std::vector<float> Simple2Agent::Thetas(
    const VectorXf& x) const {
  return {std::atan2(x(kP1VyIdx), x(kP1VxIdx)),
          std::atan2(x(kP2VyIdx), x(kP2VxIdx))};
}

}  // namespace ilqgames
