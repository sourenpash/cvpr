"""Shared fixtures for the R1 + Dex3 tests (CPU only; no Isaac Lab)."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import types

import pytest

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / "gear_sonic/data/assets/robot_description"
R1_URDF = ASSETS / "urdf/r1/r1_dex3.urdf"
R1_MJCF = ASSETS / "mjcf/r1_dex3.xml"
G1_URDF = ASSETS / "urdf/g1/main.urdf"
G1_MJCF = ASSETS / "mjcf/g1_29dof_rev_1_0.xml"
H2_URDF = ASSETS / "urdf/h2/h2.urdf"
H2_MJCF = ASSETS / "mjcf/h2.xml"
PRESET = REPO / "gear_sonic/config/exp/manager/universal_token/all_modes/sonic_r1_dex3.yaml"


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO


def load_module_constants(path: Path, prefix: str) -> dict:
    """Read top-level ``PREFIX_*`` literal assignments from a Python file without importing it.

    Needed because ``robots/g1.py`` etc. import Isaac Lab at module level.
    """
    tree = ast.parse(path.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.startswith(prefix):
                try:
                    out[name] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return out


@pytest.fixture(scope="session")
def isaaclab_stubs():
    """Install minimal stand-ins for the Isaac Lab classes used by ``robots/r1.py``.

    Lets the robot config module be imported and its dict-building logic exercised on a
    machine without Isaac Sim. Only the attribute names used by ``r1.py``/``g1.py`` exist.
    """
    if "isaaclab" in sys.modules and not getattr(sys.modules["isaaclab"], "_sonic_stub", False):
        yield  # real Isaac Lab present
        return

    class _Cfg:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def replace(self, **kw):
            new = type(self)(**self.__dict__)
            new.__dict__.update(kw)
            return new

    class ImplicitActuatorCfg(_Cfg):
        pass

    class ArticulationCfg(_Cfg):
        class InitialStateCfg(_Cfg):
            pass

    class _Sim(types.ModuleType):
        class UrdfFileCfg(_Cfg):
            pass

        class RigidBodyPropertiesCfg(_Cfg):
            pass

        class ArticulationRootPropertiesCfg(_Cfg):
            pass

        class UrdfConverterCfg(_Cfg):
            class JointDriveCfg(_Cfg):
                class PDGainsCfg(_Cfg):
                    pass

    isaaclab = types.ModuleType("isaaclab")
    isaaclab._sonic_stub = True
    actuators = types.ModuleType("isaaclab.actuators")
    actuators.ImplicitActuatorCfg = ImplicitActuatorCfg
    assets = types.ModuleType("isaaclab.assets")
    articulation = types.ModuleType("isaaclab.assets.articulation")
    articulation.ArticulationCfg = ArticulationCfg
    sim = _Sim("isaaclab.sim")
    mods = {
        "isaaclab": isaaclab,
        "isaaclab.actuators": actuators,
        "isaaclab.assets": assets,
        "isaaclab.assets.articulation": articulation,
        "isaaclab.sim": sim,
    }
    saved = {k: sys.modules.get(k) for k in mods}
    sys.modules.update(mods)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        for k in list(sys.modules):
            if k.startswith("gear_sonic.envs.manager_env.robots"):
                sys.modules.pop(k)
