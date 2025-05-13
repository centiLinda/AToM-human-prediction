#ifndef POST_PROCESSING_H
#define POST_PROCESSING_H

#include <eigen3/Eigen/Dense>
#include <cmath>
#include <stdio.h>

namespace ilqgames {

struct TrajectoryProcessor {
    Eigen::MatrixXf processTrajectory(const Eigen::MatrixXf& trajectory, 
                                      const Eigen::Vector2f& goal_1, 
                                      const Eigen::Vector2f& goal_2, 
                                      const Eigen::Vector2f& goal_3,
                                      float v1_limit, float v2_limit, float v3_limit);

private:
    Eigen::MatrixXf fixGoalPositions(Eigen::MatrixXf trajectory, const Eigen::Vector2f& goal_1, 
                                     const Eigen::Vector2f& goal_2, const Eigen::Vector2f& goal_3);
    Eigen::MatrixXf limitSpeeds(Eigen::MatrixXf trajectory, float v1_limit, float v2_limit, float v3_limit);
    bool isDivergent(const Eigen::MatrixXf& trajectory);
};

Eigen::MatrixXf TrajectoryProcessor::processTrajectory(const Eigen::MatrixXf& trajectory, 
                                                       const Eigen::Vector2f& goal_1, 
                                                       const Eigen::Vector2f& goal_2, 
                                                       const Eigen::Vector2f& goal_3, 
                                                       float v1_limit, float v2_limit, float v3_limit) {
    Eigen::MatrixXf new_trajectory = trajectory;
    
    new_trajectory = limitSpeeds(new_trajectory, v1_limit, v2_limit, v3_limit);
    new_trajectory = fixGoalPositions(new_trajectory, goal_1, goal_2, goal_3);

    return new_trajectory;
}

// Purpose is to 'drag' diverged ending points back to closest point to goal,
// all following points also move to closest point
Eigen::MatrixXf TrajectoryProcessor::fixGoalPositions(Eigen::MatrixXf trajectory, 
                                                      const Eigen::Vector2f& goal_1, 
                                                      const Eigen::Vector2f& goal_2,
                                                      const Eigen::Vector2f& goal_3) {
    int closest_1_idx = 0;
    int closest_2_idx = 0;
    int closest_3_idx = 0;
    float min_dist_1 = std::numeric_limits<float>::max();
    float min_dist_2 = std::numeric_limits<float>::max();
    float min_dist_3 = std::numeric_limits<float>::max();

    for (int i = 0; i < trajectory.rows(); ++i) {
        float dist_1 = (trajectory(i, 0) - goal_1(0)) * (trajectory(i, 0) - goal_1(0)) + (trajectory(i, 1) - goal_1(1)) * (trajectory(i, 1) - goal_1(1));
        float dist_2 = (trajectory(i, 4) - goal_2(0)) * (trajectory(i, 4) - goal_2(0)) + (trajectory(i, 5) - goal_2(1)) * (trajectory(i, 5) - goal_2(1));
        float dist_3 = (trajectory(i, 8) - goal_3(0)) * (trajectory(i, 8) - goal_3(0)) + (trajectory(i, 9) - goal_3(1)) * (trajectory(i, 9) - goal_3(1));

        if (dist_1 < min_dist_1) {
            min_dist_1 = dist_1;
            closest_1_idx = i;
        }
        if (dist_2 < min_dist_2) {
            min_dist_2 = dist_2;
            closest_2_idx = i;
        }
        if (dist_3 < min_dist_3) {
            min_dist_3 = dist_3;
            closest_3_idx = i;
        }
    }

    for (int i = closest_1_idx + 1; i < trajectory.rows(); ++i) {
        trajectory.row(i).head<2>() = trajectory.row(closest_1_idx).head<2>(); // pull to closest point
    }

    for (int i = closest_2_idx + 1; i < trajectory.rows(); ++i) {
        trajectory.row(i).segment<2>(4) = trajectory.row(closest_2_idx).segment<2>(4);
    }

    for (int i = closest_3_idx + 1; i < trajectory.rows(); ++i) {
        trajectory.row(i).segment<2>(8) = trajectory.row(closest_3_idx).segment<2>(8);
    }

    return trajectory;
}

// Purpose is re-position points on the original trajectory, to adjust the speeds
Eigen::MatrixXf TrajectoryProcessor::limitSpeeds(Eigen::MatrixXf trajectory, 
                                                 float v1_limit, 
                                                 float v2_limit,
                                                 float v3_limit) {
    int num_timesteps = trajectory.rows();
    Eigen::MatrixXf traj_new = Eigen::MatrixXf::Zero(num_timesteps, 12);
    traj_new.row(0) = trajectory.row(0);

    for (int agent = 0; agent < 3; ++agent) {
        int pos_index = agent * 4;
        int vel_index = pos_index + 2;

        int traj_new_idx = 1;
        float speed_limit;
        switch (agent) {
            case 0:
                speed_limit = v1_limit;
                break;
            case 1:
                speed_limit = v2_limit;
                break;
            case 2:
                speed_limit = v3_limit;
                break;
        }

        float threshold = 0.1;

        for (int i = 1; i < num_timesteps; ++i) {
            if (traj_new_idx >= num_timesteps) break; // do not exceed original length

            Eigen::Vector2f p_prev = traj_new.row(traj_new_idx - 1).segment<2>(pos_index);
            Eigen::Vector2f p_curr = trajectory.row(i).segment<2>(pos_index);
            float dist = (p_curr - p_prev).norm();

            // if acceptable
            if (dist >= (speed_limit - threshold) && dist <= (speed_limit + threshold)) {
                traj_new.row(traj_new_idx).segment<2>(pos_index) = p_curr;
                ++traj_new_idx;
            } 
            // if too large, search in this section
            else if (dist > (speed_limit + threshold)) {
                Eigen::Vector2f p_new = p_prev + (p_curr - p_prev).normalized() * speed_limit;
                traj_new.row(traj_new_idx).segment<2>(pos_index) = p_new;
                ++traj_new_idx;
                --i; // Re-examine the same p_curr for the next iteration
            } 
            // if too short, skip and search in the next section
            else {
                bool point_found = false;
                for (int j = i + 1; j < num_timesteps; ++j) {
                    Eigen::Vector2f p_next = trajectory.row(j).segment<2>(pos_index);
                    float next_dist = (p_next - p_prev).norm();
                    if (next_dist > speed_limit) {
                        Eigen::Vector2f p_new = p_prev + (p_next - p_prev).normalized() * speed_limit;
                        traj_new.row(traj_new_idx).segment<2>(pos_index) = p_new;
                        ++traj_new_idx;
                        i = j - 1; // Update i to the last checked position
                        point_found = true;
                        break;
                    }
                }
                if (!point_found) { // save ending point
                    traj_new.row(traj_new_idx).segment<2>(pos_index) = trajectory.row(num_timesteps - 1).segment<2>(pos_index);
                    ++traj_new_idx;
                    break;
                }     
            }
        }
        // If length if not enough
        while (traj_new_idx < num_timesteps) {
            traj_new.row(traj_new_idx).segment<2>(pos_index) = traj_new.row(traj_new_idx - 1).segment<2>(pos_index);
            ++traj_new_idx;
        }
    }

    // Recalculate velocities
    for (int agent = 0; agent < 3; ++agent) {
        int pos_index = agent * 4;
        int vel_index = pos_index + 2;

        for (int i = 1; i < num_timesteps; ++i) {
            Eigen::Vector2f p_prev = traj_new.row(i - 1).segment<2>(pos_index);
            Eigen::Vector2f p_curr = traj_new.row(i).segment<2>(pos_index);
            Eigen::Vector2f vel_prev = traj_new.row(i - 1).segment<2>(vel_index);

            Eigen::Vector2f vel_curr = (p_curr - p_prev) * 2.0f - vel_prev;
            traj_new.row(i).segment<2>(vel_index) = vel_curr;
        }
    }

    return traj_new;
}

}

#endif // POST_PROCESSING_H
