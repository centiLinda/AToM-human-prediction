//////////////////////////////////////////////////////
//
// Class for applying a given cost weighted by time,
// later timesteps are given higher weight.
//
//////////////////////////////////////////////////////

#ifndef ILQGAMES_COST_TIME_WEIGHTED_COST_H
#define ILQGAMES_COST_TIME_WEIGHTED_COST_H

#include <ilqgames/cost/cost.h>
#include <ilqgames/utils/types.h>

#include <glog/logging.h>
#include <memory>
#include <string>

namespace ilqgames {

class TimeWeightedCost : public Cost {
 public:
  ~TimeWeightedCost() {}
  TimeWeightedCost(const std::shared_ptr<const Cost>& cost, Time threshold_time, const std::string& name = "")
      : Cost(0.0, name), cost_(cost), threshold_time_(threshold_time) {
    CHECK_NOTNULL(cost.get());
  }

  // Evaluate this cost at the current time and input.
  float Evaluate(Time t, const VectorXf& input) const {
    float weight = (t - initial_time_ + 1e-4) * 0.1;
    return cost_->Evaluate(t, input) * weight;
  }

  // Quadraticize this cost at the given time and input, and add to the running
  // sum of gradients and Hessians.
  void Quadraticize(Time t, const VectorXf& input, MatrixXf* hess,
                    VectorXf* grad) const {
    float weight = (t - initial_time_ + 1e-4) * 0.1;
    
    // Temporary containers for the base cost's gradient and hessian
    MatrixXf base_hess = *hess;
    VectorXf base_grad = *grad;

    // Quadraticize the underlying cost
    cost_->Quadraticize(t, input, &base_hess, &base_grad);
    
    // Scale the gradient and the hessian by the time-dependent weight
    *grad = base_grad * weight;
    *hess = base_hess * weight;                      
  }

 private:
  // Cost function.
  const std::shared_ptr<const Cost> cost_;

  // Time threshold relative to initial time after which to apply cost.
  const Time threshold_time_;
};  //\class Cost

}  // namespace ilqgames

#endif
