#!/bin/bash

# List of nodes to kill
NODES_TO_KILL=("/viz" "/human" "/robot" "/mpc")

# Default param value
vel_1=""
detour_1=""
vel_2=""
detour_2=""

# Parse command line options
while getopts "a:b:c:d:" opt; do # can only use single alphabet
    case $opt in
        a) vel_1="$OPTARG"
        ;;
        b) detour_1="$OPTARG"
        ;;
        c) vel_2="$OPTARG"
        ;;
        d) detour_2="$OPTARG"
        ;;
        \?) echo "Invalid option -$OPTARG" >&2
            exit 1
        ;;
    esac
done

set_ros_parameters() {
    if [ ! -z "$vel_1" ]; then
        echo "Modifying config to set vel_1 to $vel_1..."
        # Using sed to replace the parameter in the YAML file
        sed -i "s/vel_1: .*/vel_1: $vel_1/" $(rospack find ilq_solver)/config/2_3AgentOvertake_config.yaml
    fi

    if [ ! -z "$detour_1" ]; then
        echo "Modifying config to set detour_1 to $detour_1..."
        # Using sed to replace the parameter in the YAML file
        sed -i "s/detour_1: .*/detour_1: $detour_1/" $(rospack find ilq_solver)/config/2_3AgentOvertake_config.yaml
    fi

    if [ ! -z "$vel_2" ]; then
        echo "Modifying config to set vel_2 to $vel_2..."
        # Using sed to replace the parameter in the YAML file
        sed -i "s/vel_2: .*/vel_2: $vel_2/" $(rospack find ilq_solver)/config/2_3AgentOvertake_config.yaml
    fi

    if [ ! -z "$detour_2" ]; then
        echo "Modifying config to set detour_2 to $detour_2..."
        # Using sed to replace the parameter in the YAML file
        sed -i "s/detour_2: .*/detour_2: $detour_2/" $(rospack find ilq_solver)/config/2_3AgentOvertake_config.yaml
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
    roslaunch simu_2 restart.launch
}

# Main execution flow
kill_nodes
check_nodes_shutdown
set_ros_parameters
relaunch_nodes