"""The ordering-map generator must reproduce the constants NVIDIA ships for G1 and H2.

If this test fails after an upstream update, the Isaac Lab ordering rule changed and
``r1_ordering.py`` must be regenerated and re-verified on a machine with Isaac Lab.
"""

import pytest

from gear_sonic.tests.r1.conftest import (
    G1_MJCF,
    G1_URDF,
    H2_MJCF,
    H2_URDF,
    REPO,
    load_module_constants,
)
from gear_sonic.utils.embodiment.ordering import compute_ordering_maps

ROBOTS = REPO / "gear_sonic/envs/manager_env/robots"


@pytest.mark.parametrize(
    "prefix,urdf,mjcf,pyfile,num_dof",
    [
        ("G1", G1_URDF, G1_MJCF, ROBOTS / "g1.py", 29),
        ("H2", H2_URDF, H2_MJCF, ROBOTS / "h2.py", 31),
    ],
)
def test_generator_reproduces_shipped_maps(prefix, urdf, mjcf, pyfile, num_dof):
    maps = compute_ordering_maps(urdf, mjcf)
    shipped = load_module_constants(pyfile, prefix)
    assert len(maps.isaaclab_dof_names) == num_dof
    for key, value in maps.as_dict().items():
        assert (
            value == shipped[f"{prefix}_{key.upper()}"]
        ), f"{prefix} {key} differs from shipped constants"


def test_maps_are_inverse_permutations():
    maps = compute_ordering_maps(G1_URDF, G1_MJCF)
    n = len(maps.isaaclab_dof_names)
    assert sorted(maps.isaaclab_to_mujoco_dof) == list(range(n))
    for mj_i, il_i in enumerate(maps.isaaclab_to_mujoco_dof):
        assert maps.mujoco_to_isaaclab_dof[il_i] == mj_i
    nb = len(maps.isaaclab_joints)
    assert sorted(maps.isaaclab_to_mujoco_body) == list(range(nb))
    for mj_i, il_i in enumerate(maps.isaaclab_to_mujoco_body):
        assert maps.mujoco_to_isaaclab_body[il_i] == mj_i
