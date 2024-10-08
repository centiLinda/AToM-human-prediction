#ifndef POST_PROCESSING_H
#define POST_PROCESSING_H

#include <eigen3/Eigen/Dense>
#include <cmath>
#include <stdio.h>

namespace ilqgames {

struct TrajectoryProcessor {
    Eigen::MatrixXf processTrajectory(const Eigen::MatrixXf& trajectory, float v1_limit, float v2_limit);

private:
    Eigen::MatrixXf limitSpeeds(Eigen::MatrixXf trajectory, float v1_limit, float v2_limit);
};

Eigen::MatrixXf TrajectoryProcessor::processTrajectory(const Eigen::MatrixXf& trajectory, float v1_limit, float v2_limit) {
    Eigen::MatrixXf new_trajectory = trajectory;
    
    new_trajectory = limitSpeeds(new_trajectory, v1_limit, v2_limit);

    return new_trajectory;
}

// Purpose is re-position points on the original trajectory, to constrain the speeds
Eigen::MatrixXf TrajectoryProcessor::limitSpeeds(Eigen::MatrixXf trajectory, float v1_limit, float v2_limit) {
    int num_timesteps = trajectory.rows();
    Eigen::MatrixXf traj_new = Eigen::MatrixXf::Zero(num_timesteps, 8);
    traj_new.row(0) = trajectory.row(0);

    for (int agent = 0; agent < 2; ++agent) {
        int pos_index = agent * 4;
        int vel_index = pos_index + 2;

        int traj_new_idx = 1;
        float speed_limit = (agent == 0) ? v1_limit : v2_limit;
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
    for (int agent = 0; agent < 2; ++agent) {
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