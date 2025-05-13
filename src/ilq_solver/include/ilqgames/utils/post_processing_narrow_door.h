#ifndef POST_PROCESSING_H
#define POST_PROCESSING_H

#include <eigen3/Eigen/Dense>
#include <cmath>
#include <stdio.h>

namespace ilqgames {

struct TrajectoryProcessor {
    Eigen::MatrixXf processTrajectory(const Eigen::MatrixXf& trajectory, 
                                      float v1_limit, float v2_limit);

private:
    Eigen::MatrixXf limitSpeeds(Eigen::MatrixXf trajectory, float v1_limit, float v2_limit);
    Eigen::MatrixXf stopAtWall(Eigen::MatrixXf trajectory, float wall_x, float wall_y, float wall_width, float wall_gap, 
                                float v1_limit, float v2_limit, float buffer);
};

Eigen::MatrixXf TrajectoryProcessor::processTrajectory(const Eigen::MatrixXf& trajectory, 
                                                       float v1_limit, float v2_limit) {
    Eigen::MatrixXf new_trajectory = trajectory;
    
    new_trajectory = limitSpeeds(new_trajectory, v1_limit, v2_limit);
    new_trajectory = stopAtWall(new_trajectory, 2.5f, 2.5f, 1.0f, 1.5f,  //TODO parse wall position instead of define here
                                v1_limit, v2_limit, -0.2);

    return new_trajectory;
}

Eigen::MatrixXf TrajectoryProcessor::stopAtWall(Eigen::MatrixXf trajectory, 
                                                float wall_x,
                                                float wall_y, // x&y center
                                                float wall_width, 
                                                float wall_gap,
                                                float v1_limit,
                                                float v2_limit,
                                                float buffer) {
    for (int agent = 0; agent < 2; ++agent) {

        int pos_index = agent * 4;
        int vel_index = pos_index + 2;
        int direction = (agent == 0) ? -1 : 1;; // -1 for agent1, 1 for agent2
        float speed_limit = (agent == 0) ? v1_limit : v2_limit;

        // start from 2nd step
        for (int i = 1; i < trajectory.rows(); ++i) {
            Eigen::Vector2f position = trajectory.row(i).segment<2>(pos_index);

            bool y_check = (agent == 0) ? (position.y() >= wall_y + wall_gap / 2.0f + buffer) : (position.y() <= wall_y - wall_gap / 2.0f - buffer);

            if (position.x() >= wall_x - wall_width / 2.0f + buffer && 
                position.x() <= wall_x && // only deal with left side of collision
                y_check) {

                // check if previous step is too far away from the gap
                Eigen::Vector2f previous_position = trajectory.row(i-1).segment<2>(pos_index);
                float diff = std::abs(previous_position(1) - wall_y) - wall_gap / 2.0f;
                if (diff > speed_limit){
                    previous_position(1) = previous_position(1) + direction * speed_limit;
                }

                // freeze all subsequent steps
                for (int j = i; j < trajectory.rows(); ++j) {
                    trajectory.row(j).segment<2>(pos_index) = previous_position;
                    trajectory.row(j).segment<2>(vel_index).setZero();
                }
                break;
            }
        }
    }
    return trajectory;
}

// Purpose is re-position points on the original trajectory, to adjust the speeds
Eigen::MatrixXf TrajectoryProcessor::limitSpeeds(Eigen::MatrixXf trajectory, 
                                                 float v1_limit, 
                                                 float v2_limit) {
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
