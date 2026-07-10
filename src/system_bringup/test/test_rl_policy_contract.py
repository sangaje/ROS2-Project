from copy import deepcopy

from gymnasium.spaces import Box, Dict
import numpy as np
import pytest

from system_bringup.rl_policy_contract import (
    PolicyContractError,
    active_scout_config,
    load_deployment_model,
    load_contract,
    probe_checkpoint,
    validate_static_assets,
)


def test_contract_compiles_the_actual_v132_in_process_configuration():
    contract = load_contract()
    assets = validate_static_assets(contract)
    config = active_scout_config(contract)

    assert assets['runner'].name == 'run_train_v132_clean.sh'
    assert assets['checkpoint'].name == 'sac_turtlebot3_burger.zip'
    assert config.checkpoint == assets['checkpoint']
    assert config.control_dt_sec == 0.2
    assert config.map_substeps_per_action == 2
    assert config.lidar_bins == 60
    assert config.map_obs_size == 64
    assert config.action_high == (0.22, 0.7)
    assert config.safety_trigger_distance_m == 0.22
    assert config.safety_slow_distance_m == 0.4


def test_contract_rejects_non_v132_map_substep_rate():
    contract = deepcopy(load_contract())
    contract['runtime']['map_substeps_per_action'] = 1

    with pytest.raises(PolicyContractError, match='map_substeps_per_action'):
        active_scout_config(contract)


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


def test_deployment_checkpoint_predicts_continuously_with_legacy_numpy_pickle():
    """The deployed NumPy-2 checkpoint must run on the robot's NumPy-1 image."""
    model = load_deployment_model()
    observation = {
        'map': np.zeros((4, 64, 64), dtype=np.float32),
        'map_seq': np.zeros((8, 4, 64, 64), dtype=np.float32),
        'seq': np.zeros((8, 69), dtype=np.float32),
        'vector': np.zeros((69,), dtype=np.float32),
    }

    actions = [model.predict(observation, deterministic=True)[0] for _ in range(20)]

    assert len(actions) == 20
    assert all(np.asarray(action).shape == (2,) for action in actions)
    assert all(np.all(np.isfinite(action)) for action in actions)
