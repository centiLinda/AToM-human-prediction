#!/bin/bash

# List of nodes to kill
NODES_TO_KILL=("/viz" "/human" "/robot" "/mpc")

# Default param value
human_detour=""
human_v=""

# Parse command line options
while getopts "d:v:" opt; do
    case $opt in
        d) human_detour="$OPTARG"
        ;;
        v) human_v="$OPTARG"
        ;;
        \?) echo "Invalid option -$OPTARG" >&2
            exit 1
        ;;
    esac
done

set_ros_parameters() {
    if [ ! -z "$human_detour" ]; then
        echo "Modifying config to set human_detour to $human_detour..."
        # Using sed to replace the parameter in the YAML file
        sed -i "s/human_detour: .*/human_detour: $human_detour/" $(rospack find ilq_solver)/config/1_2AgentExchange_config.yaml
    fi

    if [ ! -z "$human_v" ]; then
        echo "Modifying config to set human_v to $human_v..."
        # Using sed to replace the parameter in the YAML file
        sed -i "s/human_v: .*/human_v: $human_v/" $(rospack find ilq_solver)/config/1_2AgentExchange_config.yaml
    fi
}

# Function to kill nodes
kill_nodes() {
    echo "Killing nodes..."
    for node in "${NODES_TO_KILL[@]}"; do
        rosnode kill "$node"
    done
}

# Function to check if all nodes are shutdown
check_nodes_shutdown() {
    echo "Checking for node shutdown..."
    for node in "${NODES_TO_KILL[@]}"; do
        while rosnode list | grep -q "$node"; do
            echo "Waiting for $node to shutdown..."
            sleep 0.5
        done
    done
    echo "All nodes have been shut down."
}

# Function to relaunch nodes
relaunch_nodes() {
    echo "Relaunching nodes..."
    roslaunch simu_1 restart.launch
}

# Main execution flow
kill_nodes
check_nodes_shutdown
set_ros_parameters
relaunch_nodes