#ifndef KALMAN_CORRIDOR_3_AGENT_HPP_
#define KALMAN_CORRIDOR_3_AGENT_HPP_

#include <kalman/LinearizedSystemModel.hpp>
#include <kalman/LinearizedMeasurementModel.hpp>
#include <ilqgames/ros/corridor_3_agent.h>
#include <ilqgames/utils/post_processing_3_agent.h>
#include <random>

// Process State & Model
namespace ukfGame
{
template<typename T> // numeric scalar type
class State : public Kalman::Vector<T, 6> // Dimension = number of params we wish to update
{
public:
    KALMAN_VECTOR(State, T, 6)
    
    static constexpr size_t prox1_thresholdId = 0;
    static constexpr size_t prox2_thresholdId = 1;
    static constexpr size_t prox3_thresholdId = 2;
    static constexpr size_t v1_limitId = 3;
    static constexpr size_t v2_limitId = 4;
    static constexpr size_t v3_limitId = 5;
    
    T prox1_threshold()     const { return (*this)[ prox1_thresholdId ]; }
    T prox2_threshold()     const { return (*this)[ prox2_thresholdId ]; }
    T prox3_threshold()     const { return (*this)[ prox3_thresholdId ]; }
    T v1_limit()     const { return (*this)[ v1_limitId ]; }
    T v2_limit()     const { return (*this)[ v2_limitId ]; }
    T v3_limit()     const { return (*this)[ v3_limitId ]; }
    
    T& prox1_threshold()    { return (*this)[ prox1_thresholdId ]; }
    T& prox2_threshold()    { return (*this)[ prox2_thresholdId ]; }
    T& prox3_threshold()    { return (*this)[ prox3_thresholdId ]; }
    T& v1_limit()    { return (*this)[ v1_limitId ]; }
    T& v2_limit()    { return (*this)[ v2_limitId ]; }
    T& v3_limit()    { return (*this)[ v3_limitId ]; }
};

template<typename T, template<class> class CovarianceBase = Kalman::StandardBase>
class ProcessModel : public Kalman::LinearizedSystemModel<State<T>, Kalman::Vector<T, 0>, CovarianceBase>
{
public:
	typedef ukfGame::State<T> S;
    typedef Kalman::Vector<T, 0> C; // Dummy control type

    S f(const S& x, const C& /*u*/) const override
    {
        S x_;

        x_.prox1_threshold() = x.prox1_threshold();
        x_.prox2_threshold() = x.prox2_threshold();
        x_.prox3_threshold() = x.prox3_threshold();
        x_.v1_limit() = x.v1_limit();
        x_.v2_limit() = x.v2_limit();
        x_.v3_limit() = x.v3_limit();
        
        return x_;
    }

};

// Measurement State & Model
template<typename T>
class Measurement : public Kalman::Vector<T, 12>
{
public:
    KALMAN_VECTOR(Measurement, T, 12)
    
    static constexpr size_t x1Id = 0;
    static constexpr size_t y1Id = 1;
    static constexpr size_t vx1Id = 2;
    static constexpr size_t vy1Id = 3;
    static constexpr size_t x2Id = 4;
    static constexpr size_t y2Id = 5;
    static constexpr size_t vx2Id = 6;
    static constexpr size_t vy2Id = 7;
    static constexpr size_t x3Id = 8;
    static constexpr size_t y3Id = 9;
    static constexpr size_t vx3Id = 10;
    static constexpr size_t vy3Id = 11;
    
    T x1()   const { return (*this)[ x1Id ]; }
    T y1()   const { return (*this)[ y1Id ]; }
    T vx1()  const { return (*this)[ vx1Id ]; }
    T vy1()  const { return (*this)[ vy1Id ]; }
    T x2()   const { return (*this)[ x2Id ]; }
    T y2()   const { return (*this)[ y2Id ]; }
    T vx2()  const { return (*this)[ vx2Id ]; }
    T vy2()  const { return (*this)[ vy2Id ]; }
    T x3()   const { return (*this)[ x3Id ]; }
    T y3()   const { return (*this)[ y3Id ]; }
    T vx3()  const { return (*this)[ vx3Id ]; }
    T vy3()  const { return (*this)[ vy3Id ]; }
    
    T& x1()  { return (*this)[ x1Id ]; }
    T& y1()  { return (*this)[ y1Id ]; }
    T& vx1() { return (*this)[ vx1Id ]; }
    T& vy1() { return (*this)[ vy1Id ]; }
    T& x2()  { return (*this)[ x2Id ]; }
    T& y2()  { return (*this)[ y2Id ]; }
    T& vx2() { return (*this)[ vx2Id ]; }
    T& vy2() { return (*this)[ vy2Id ]; }
    T& x3()  { return (*this)[ x3Id ]; }
    T& y3()  { return (*this)[ y3Id ]; }
    T& vx3() { return (*this)[ vx3Id ]; }
    T& vy3() { return (*this)[ vy3Id ]; }
};

template<typename T, template<class> class CovarianceBase = Kalman::StandardBase>
class MeasurementModel : public Kalman::LinearizedMeasurementModel<State<T>, Measurement<T>, CovarianceBase>
{
public:
    typedef  ukfGame::State<T> S;
    typedef  ukfGame::Measurement<T> M;
    
    MeasurementModel(ilqgames::SolverParams& params,
                     ilqgames::Corridor3AgentConfig& config,
                     std::shared_ptr<ilqgames::Corridor3Agent>& problem):
                     params_(params), config_(config), problem_(problem)
    {
        // this->V.setIdentity();
    }
    
    M h(const S& x) const
    {
        M measurement;

        ilqgames::Corridor3AgentConfig localConfig = config_;
        localConfig.prox1_threshold = x.prox1_threshold();
        localConfig.prox2_threshold = x.prox2_threshold();
        localConfig.prox3_threshold = x.prox3_threshold();
        localConfig.v1_limit = x.v1_limit();
        localConfig.v2_limit = x.v2_limit();
        localConfig.v3_limit = x.v3_limit();
        problem_->SetupConfig(localConfig);
        ilqgames::AugmentedLagrangianSolver solver(problem_, params_);
        std::shared_ptr<const ilqgames::SolverLog> log = solver.Solve();
        size_t iteration = log->NumIterates();
        Eigen::MatrixXf results = log->GetPrediction(iteration-1);

        // -----------------------------Post-processing-----------------------------------
        Eigen::Vector2f goal_1(localConfig.x1_goal, localConfig.y1_goal);
        Eigen::Vector2f goal_2(localConfig.x2_goal, localConfig.y2_goal);
        Eigen::Vector2f goal_3(localConfig.x3_goal, localConfig.y3_goal);
        ilqgames::TrajectoryProcessor processor;
        Eigen::MatrixXf processed_trajectory = processor.processTrajectory(results, goal_1, goal_2, goal_3,
                                                                           localConfig.v1_limit, localConfig.v2_limit, localConfig.v3_limit);
        results = processed_trajectory;

        measurement.x1() = results(1, 0);
        measurement.y1() = results(1, 1);
        measurement.vx1() = results(1, 2);
        measurement.vy1() = results(1, 3);
        measurement.x2() = results(1, 4);
        measurement.y2() = results(1, 5);
        measurement.vx2() = results(1, 6);
        measurement.vy2() = results(1, 7);
        measurement.x3() = results(1, 8);
        measurement.y3() = results(1, 9);
        measurement.vx3() = results(1, 10);
        measurement.vy3() = results(1, 11);

        return measurement;
    }
    
protected:
    ilqgames::SolverParams& params_;
    ilqgames::Corridor3AgentConfig& config_;
    std::shared_ptr<ilqgames::Corridor3Agent>& problem_;
};

} // namespace ukfGame

#endif