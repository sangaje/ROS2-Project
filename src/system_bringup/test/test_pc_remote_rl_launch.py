import importlib.util
from pathlib import Path


def _load_module():
    launch_path = (
        Path(__file__).resolve().parents[1] / 'launch' / 'pc_remote_rl.launch.py'
    )
    spec = importlib.util.spec_from_file_location('pc_remote_rl_launch', launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_loads_and_declares_only_domain_id():
    module = _load_module()

    description = module.generate_launch_description()

    arg_names = {
        entity.name
        for entity in description.entities
        if type(entity).__name__ == 'DeclareLaunchArgument'
    }
    assert arg_names == {'domain_id'}
