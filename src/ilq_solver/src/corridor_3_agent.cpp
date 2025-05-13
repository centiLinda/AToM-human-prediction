#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/cost/semiquadratic_norm_cost.h>
#include <ilqgames/cost/proximity_cost.h>
#include <ilqgames/cost/final_time_cost.h>
#include <ilqgames/cost/time_weighted_cost.h>
#include <ilqgames/cost/semiquadratic_polyline2_cost.h>
#include <ilqgames/constraint/single_dimension_constraint.h>

#include <ilqgames/dynamics/concatenated_dynamical_system.h>
#include <ilqgames/dynamics/single_player_point_mass_2d.h>

#include <ilqgames/geometry/polyline2.h>

#include <ilqgames/ros/corridor_3_agent.h>

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

static const Dimension kP3PxIdx = Dyn::kNumXDims + Dyn::kNumXDims + Dyn::kPxIdx;
static const Dimension kP3PyIdx = Dyn::kNumXDims + Dyn::kNumXDims + Dyn::kPyIdx;
static const Dimension kP3VxIdx = Dyn::kNumXDims + Dyn::kNumXDims + Dyn::kVxIdx;
static const Dimension kP3VyIdx = Dyn::kNumXDims + Dyn::kNumXDims + Dyn::kVyIdx;
}  // anonymous namespace

void Corridor3Agent::ConstructDynamics() {
  dynamics_.reset(new ConcatenatedDynamicalSystem(
      {std::make_shared<Dyn>(), 
       std::make_shared<Dyn>(),
       std::make_shared<Dyn>()}));
}

void Corridor3Agent::ConstructInitialState() {
  x0_ = VectorXf::Zero(dynamics_->XDim());

  x0_(kP1PxIdx) = this->x1_initial;
  x0_(kP1PyIdx) = this->y1_initial;
  x0_(kP1VxIdx) = this->vx1_initial;
  x0_(kP1VyIdx) = this->vy1_initial;

  x0_(kP2PxIdx) = this->x2_initial;
  x0_(kP2PyIdx) = this->y2_initial;
  x0_(kP2VxIdx) = this->vx2_initial;
  x0_(kP2VyIdx) = this->vy2_initial;

  x0_(kP3PxIdx) = this->x3_initial;
  x0_(kP3PyIdx) = this->y3_initial;
  x0_(kP3VxIdx) = this->vx3_initial;
  x0_(kP3VyIdx) = this->vy3_initial;
}

void Corridor3Agent::ConstructPlayerCosts() {
  // Set up costs for all players.
  player_costs_.emplace_back("P1", 1.0, 0.0);
  player_costs_.emplace_back("P2", 1.0, 0.0);
  player_costs_.emplace_back("P3", 1.0, 0.0);
  auto& p1_cost = player_costs_[0];
  auto& p2_cost = player_costs_[1];
  auto& p3_cost = player_costs_[2];

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

  const auto p3_goalx_cost = std::make_shared<TimeWeightedCost>(
    std::make_shared<QuadraticCost>(this->wGoal, kP3PxIdx, this->x3_goal), 
    time::kTimeHorizon - FinalTimeWindow, "GoalX3");
  const auto p3_goaly_cost = std::make_shared<TimeWeightedCost>(
    std::make_shared<QuadraticCost>(this->wGoal, kP3PyIdx, this->y3_goal), 
    time::kTimeHorizon - FinalTimeWindow, "GoalY3");
  p3_cost.AddStateCost(p3_goalx_cost);
  p3_cost.AddStateCost(p3_goaly_cost);

  // 2. speed cost
  const std::shared_ptr<SemiquadraticNormCost> p1_v_cost(
    new SemiquadraticNormCost(this->wSpeed, {kP1VxIdx, kP1VyIdx}, this->v1_limit, true, "Max_Speed1"));
  const std::shared_ptr<SemiquadraticNormCost> p2_v_cost(
    new SemiquadraticNormCost(this->wSpeed, {kP2VxIdx, kP2VyIdx}, this->v2_limit, true, "Max_Speed2"));
  const std::shared_ptr<SemiquadraticNormCost> p3_v_cost(
    new SemiquadraticNormCost(this->wSpeed, {kP3VxIdx, kP3VyIdx}, this->v3_limit, true, "Max_Speed3"));
  p1_cost.AddStateCost(p1_v_cost);
  p2_cost.AddStateCost(p2_v_cost);
  p3_cost.AddStateCost(p3_v_cost);

  // 3. proximity cost
  const std::shared_ptr<ProximityCost> p1_p2_proximity_cost(
    new ProximityCost(this->wProximity, {kP1PxIdx, kP1PyIdx}, {kP2PxIdx, kP2PyIdx}, this->prox1_threshold, "ProximityP1p2"));
  const std::shared_ptr<ProximityCost> p1_p3_proximity_cost(
    new ProximityCost(this->wProximity, {kP1PxIdx, kP1PyIdx}, {kP3PxIdx, kP3PyIdx}, this->prox1_threshold, "ProximityP1p3"));
  p1_cost.AddStateCost(p1_p2_proximity_cost);
  p1_cost.AddStateCost(p1_p3_proximity_cost);

  const std::shared_ptr<ProximityCost> p2_p1_proximity_cost(
    new ProximityCost(this->wProximity, {kP2PxIdx, kP2PyIdx}, {kP1PxIdx, kP1PyIdx}, this->prox2_threshold, "ProximityP2p1"));
  const std::shared_ptr<ProximityCost> p2_p3_proximity_cost(
    new ProximityCost(this->wProximity, {kP2PxIdx, kP2PyIdx}, {kP3PxIdx, kP3PyIdx}, this->prox2_threshold, "ProximityP2p3"));
  p2_cost.AddStateCost(p2_p1_proximity_cost);
  p2_cost.AddStateCost(p2_p3_proximity_cost);

  const std::shared_ptr<ProximityCost> p3_p2_proximity_cost(
    new ProximityCost(this->wProximity, {kP3PxIdx, kP3PyIdx}, {kP2PxIdx, kP2PyIdx}, this->prox3_threshold, "ProximityP3p2"));
  const std::shared_ptr<ProximityCost> p3_p1_proximity_cost(
    new ProximityCost(this->wProximity, {kP3PxIdx, kP3PyIdx}, {kP1PxIdx, kP1PyIdx}, this->prox3_threshold, "ProximityP3p1"));
  p3_cost.AddStateCost(p3_p2_proximity_cost);
  p3_cost.AddStateCost(p3_p1_proximity_cost);

  // 4. control cost
  const auto control_cost = std::make_shared<QuadraticCost>(this->wControl, -1, 0.0, "ControlCost");
  p1_cost.AddControlCost(0, control_cost);
  p2_cost.AddControlCost(1, control_cost);
  p3_cost.AddControlCost(2, control_cost);

  // 5. obstacle cost
  const Polyline2 lane1({Point2(0.0, 0.1), Point2(0.1, this->end_y)}); // left wall
  const Polyline2 lane2({Point2(this->right_x, 0.1), Point2(this->right_x, this->end_y)}); // right wall
  //#### wall starting_y cannot be 0, raise small number error
  
  const std::shared_ptr<SemiquadraticPolyline2Cost> p1_left_r_cost(
      new SemiquadraticPolyline2Cost(this->wObstacle, lane1,
                                     {kP1PxIdx, kP1PyIdx}, this->width,
                                     true, "LeftWallRightBoundary"));
  const std::shared_ptr<SemiquadraticPolyline2Cost> p2_right_l_cost(
      new SemiquadraticPolyline2Cost(this->wObstacle, lane2,
                                     {kP2PxIdx, kP2PyIdx}, -this->width,
                                     false, "RightWallLeftBoundary"));
  const std::shared_ptr<SemiquadraticPolyline2Cost> p3_right_l_cost(
      new SemiquadraticPolyline2Cost(this->wObstacle, lane2,
                                     {kP3PxIdx, kP3PyIdx}, -this->width,
                                     false, "RightWallLeftBoundary"));
  
  p1_cost.AddStateCost(p1_left_r_cost);
  p2_cost.AddStateCost(p2_right_l_cost);
  p3_cost.AddStateCost(p3_right_l_cost);
}

inline std::vector<float> Corridor3Agent::Xs(const VectorXf& x) const {
  return {x(kP1PxIdx), x(kP2PxIdx), x(kP3PxIdx)};
}

inline std::vector<float> Corridor3Agent::Ys(const VectorXf& x) const {
  return {x(kP1PyIdx), x(kP2PyIdx), x(kP3PyIdx)};
}

inline std::vector<float> Corridor3Agent::Thetas(
    const VectorXf& x) const {
  return {std::atan2(x(kP1VyIdx), x(kP1VxIdx)),
          std::atan2(x(kP2VyIdx), x(kP2VxIdx)),
          std::atan2(x(kP3VyIdx), x(kP3VxIdx))};
}

}  // namespace ilqgames
