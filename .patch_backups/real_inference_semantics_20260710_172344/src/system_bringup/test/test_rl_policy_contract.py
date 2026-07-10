from copy import deepcopy

from gymnasium.spaces import Box, Dict
import numpy as np
import pytest

from system_bringup.rl_policy_contract import (
    inference_command,
    load_contract,
    PolicyContractError,
    probe_checkpoint,
    run_checkpoint_preflight,
    validate_static_assets,
)


def test_contract_assets_and_command_match_actual_v132_checkpoint():
    contract = load_contract()
    assets = validate_static_assets(contract)
    command = inference_command(contract)

    assert assets['runner'].name == 'run_train_v132_clean.sh'
    assert assets['checkpoint'].name == 'sac_turtlebot3_burger.zip'
    assert str(assets['python']) == '/home/seil/venv/ros/bin/python'
    assert command[0] == '/home/seil/venv/ros/bin/python'
    assert command[1:3] == ['-m', 'turtlebot3_rl_training.eval_policy']
    assert command[command.index('--map-obs-size') + 1] == '64'
    assert command[command.index('--num-lidar-bins') + 1] == '60'
    assert command[command.index('--max-linear-speed') + 1] == '0.22'
    assert command[command.index('--max-angular-speed') + 1] == '0.7'
    assert '--real-robot-disable-priority' in command
    assert not any(
        token.startswith('--') and 'sde' in token.lower()
        for token in command
    )


def test_checkpoint_preflight_runs_in_numpy2_venv():
    output = run_checkpoint_preflight()

    assert 'RL_POLICY_PREFLIGHT_PASS' in output
    assert '"deterministic_inference": true' in output
    assert '"sde_inference": false' in output


def test_wrong_observation_contract_fails_fast():
    contract = deepcopy(load_contract())
    contract['observation_contract']['map']['shape'] = [4, 32, 32]

    extractor_type = type('MapVectorFeatureExtractor', (), {})
    extractor_type.__module__ = 'turtlebot3_rl_training.feature_extractor'
    actor = type('Actor', (), {'features_extractor': extractor_type()})()
    model = type('Model', (), {})()
    model.observation_space = Dict({
        'map': Box(-1.0, 1.0, (4, 64, 64), dtype=np.float32),
        'map_seq': Box(-1.0, 1.0, (8, 4, 64, 64), dtype=np.float32),
        'seq': Box(-1.0, 1.0, (8, 69), dtype=np.float32),
        'vector': Box(-1.0, 1.0, (69,), dtype=np.float32),
    })
    model.action_space = Box(
        low=np.array([0.0, -0.7], dtype=np.float32),
        high=np.array([0.22, 0.7], dtype=np.float32),
        dtype=np.float32,
    )
    model.use_sde = True
    model.actor = actor

    with pytest.raises(PolicyContractError, match='observation space'):
        probe_checkpoint(contract, model=model)
