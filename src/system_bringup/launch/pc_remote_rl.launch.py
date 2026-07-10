#!/usr/bin/env python3
"""Run the ACTIVE_SCOUT RL inference process on this PC instead of the robot.

Use when the robot's own CPU can't sustain the contract's control_dt (e.g. a
Raspberry Pi struggling to hit 5 Hz). Launch the robot with
system.launch.py ... enable_exploration:=false so nothing local claims
/cmd_vel, then run this on the PC against the same ROS_DOMAIN_ID.

This does not touch DDS/CycloneDDS settings -- it inherits whatever
discovery configuration is already in this shell (ROS_AUTOMATIC_DISCOVERY_
RANGE, CYCLONEDDS_URI, etc.). If `ros2 node list` from this shell already
shows the robot's nodes, that configuration already works; only
domain_id is set here, to make sure it matches the scout.

The eval_policy command/environment are the same contract-driven ones
unified_field_robot would run locally (system_bringup.rl_policy_contract) --
nothing here re-types RL flags by hand.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration

from system_bringup.rl_policy_contract import (
    inference_command,
    inference_environment,
    run_checkpoint_preflight,
)


def generate_launch_description():
    domain_id = LaunchConfiguration('domain_id')

    def make_actions(context):
        domain_value = domain_id.perform(context).strip()
        if not domain_value:
            raise ValueError(
                'domain_id is required (defaults to $ROS_DOMAIN_ID) -- must '
                'match the scout\'s own domain, not a bridge/PC domain.'
            )

        preflight_output = run_checkpoint_preflight()

        env = {'ROS_DOMAIN_ID': domain_value}
        env.update(inference_environment())

        return [
            LogInfo(msg=[preflight_output]),
            LogInfo(msg=['PC_REMOTE_RL | domain=', domain_value]),
            ExecuteProcess(
                cmd=inference_command(),
                # Merge only ROS_DOMAIN_ID + the contract's TB3_RL_* vars
                # into this shell's environment. Deliberately does not set
                # CYCLONEDDS_URI/RMW_IMPLEMENTATION/discovery-range -- this
                # shell's own settings are what let `ros2 node list`
                # already see the robot's graph; overriding them here would
                # replace a working config with an untested one.
                additional_env=env,
                output='screen',
                name='eval_policy_remote',
            ),
        ]

    return LaunchDescription([
        DeclareLaunchArgument(
            'domain_id',
            default_value=EnvironmentVariable('ROS_DOMAIN_ID'),
            description='Must match the scout\'s own ROS_DOMAIN_ID (e.g. 22), not a bridge domain.',
        ),
        OpaqueFunction(function=make_actions),
    ])
