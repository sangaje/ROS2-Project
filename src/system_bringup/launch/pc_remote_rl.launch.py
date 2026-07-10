#!/usr/bin/env python3
"""Run the ACTIVE_SCOUT RL inference process on this PC instead of the robot.

Use when the robot's own CPU can't sustain the contract's control_dt (e.g. a
Raspberry Pi struggling to hit 5 Hz). Launch the robot with
system.launch.py ... enable_exploration:=false so nothing local claims
/cmd_vel, then run this on the PC against the same ROS_DOMAIN_ID.

Cross-machine discovery over a mesh VPN (e.g. Tailscale) does not carry
multicast, so this disables multicast and points CycloneDDS at the robot's
address directly via an explicit unicast peer -- otherwise the two sides
never find each other even on the same domain ID.

The eval_policy command/environment are the same contract-driven ones
unified_field_robot would run locally (system_bringup.rl_policy_contract) --
nothing here re-types RL flags by hand.
"""

from pathlib import Path
import tempfile

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
)
from launch.substitutions import EnvironmentVariable, LaunchConfiguration

from system_bringup.rl_policy_contract import (
    inference_command,
    inference_environment,
    run_checkpoint_preflight,
)


CYCLONEDDS_TEMPLATE = """<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain id="any">
    <General>
      <AllowMulticast>false</AllowMulticast>
    </General>
    <Discovery>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>240</MaxAutoParticipantIndex>
      <Peers>
        <Peer address="{scout_host}"/>
      </Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
"""


def _write_cyclonedds_peer_config(scout_host: str) -> Path:
    directory = Path(tempfile.gettempdir()) / 'system_bringup_pc_remote_rl'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'cyclonedds_pc_remote_rl.xml'
    path.write_text(CYCLONEDDS_TEMPLATE.format(scout_host=scout_host))
    return path


def generate_launch_description():
    scout_host = LaunchConfiguration('scout_host')
    domain_id = LaunchConfiguration('domain_id')

    def make_actions(context):
        host_value = scout_host.perform(context).strip()
        if not host_value:
            raise ValueError(
                'scout_host is required -- pass the robot\'s reachable '
                'address (Tailscale hostname or IP), e.g. '
                'scout_host:=pi2.taile3321c.ts.net'
            )
        domain_value = domain_id.perform(context).strip()
        if not domain_value:
            raise ValueError(
                'domain_id is required (defaults to $ROS_DOMAIN_ID) -- must '
                'match the scout\'s own domain, not a bridge/PC domain.'
            )

        cyclonedds_path = _write_cyclonedds_peer_config(host_value)

        preflight_output = run_checkpoint_preflight()

        env = {
            'ROS_DOMAIN_ID': domain_value,
            'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp',
            'CYCLONEDDS_URI': f'file://{cyclonedds_path}',
        }
        env.update(inference_environment())

        command = inference_command()

        return [
            LogInfo(msg=[preflight_output]),
            LogInfo(msg=[
                'PC_REMOTE_RL | domain=', domain_value,
                ' scout_host=', host_value,
                ' cyclonedds_config=', str(cyclonedds_path),
            ]),
            ExecuteProcess(
                cmd=command,
                # additional_env merges into the inherited environment.
                # env= would replace it outright and drop PYTHONPATH (from
                # `source install/setup.bash`), which is what actually makes
                # turtlebot3_rl_training importable by the venv interpreter.
                additional_env=env,
                output='screen',
                name='eval_policy_remote',
            ),
        ]

    return LaunchDescription([
        DeclareLaunchArgument(
            'scout_host',
            default_value='',
            description=(
                'Required. The scout\'s reachable network address -- '
                'Tailscale MagicDNS hostname (e.g. pi2.taile3321c.ts.net) '
                'or IP. Used as an explicit CycloneDDS unicast peer since '
                'mesh VPNs typically do not carry multicast SPDP.'
            ),
        ),
        DeclareLaunchArgument(
            'domain_id',
            default_value=EnvironmentVariable('ROS_DOMAIN_ID'),
            description='Must match the scout\'s own ROS_DOMAIN_ID (e.g. 22), not a bridge domain.',
        ),
        OpaqueFunction(function=make_actions),
    ])
