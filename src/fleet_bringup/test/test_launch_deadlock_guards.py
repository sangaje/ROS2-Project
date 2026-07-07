import ast
from pathlib import Path


LAUNCH_DIR = Path(__file__).parents[1] / 'launch'
CRITICAL_LAUNCHES = [
    LAUNCH_DIR / 'base.launch.py',
    LAUNCH_DIR / 'leader.launch.py',
    LAUNCH_DIR / 'member.launch.py',
    LAUNCH_DIR / 'follower.launch.py',
]


def _names_used(path):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def test_critical_launches_do_not_gate_on_process_exit():
    forbidden = {
        'OnProcessExit',
        'RegisterEventHandler',
        'EmitEvent',
        'Shutdown',
    }

    for path in CRITICAL_LAUNCHES:
        assert forbidden.isdisjoint(_names_used(path)), path
