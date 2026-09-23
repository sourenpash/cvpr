# Unitree R1 + Dex3-1 asset provenance

Generated files (do not hand-edit; re-run `python scripts/r1/build_r1_assets.py`):

| File | Derived from |
|---|---|
| `urdf/r1/r1_dex3.urdf` | `unitree_ros/robots/r1_description/R1.urdf` + `unitree_ros/robots/dexterous_hand_description/dex3_1/dex3_1_{l,r}.urdf` |
| `urdf/r1/meshes/*.STL` (59) | `unitree_ros/robots/r1_description/meshes/` (43) + `.../dex3_1/meshes/` (16), unmodified copies |
| `mjcf/r1_dex3.xml` | `unitree_rl_mjlab/src/assets/robots/unitree_r1/xmls/r1.xml` (+ Dex3 mass fused into wrist bodies, `<actuator>` block, meshdir, dangling `<exclude>` removed) |
| `urdf/r1/R1_DEX3_DERIVED.json` | measured from `mjcf/r1_dex3.xml` with MuJoCo |
| `gear_sonic/envs/manager_env/robots/r1_ordering.py` | computed from the URDF/MJCF pair (`gear_sonic/utils/embodiment/ordering.py`) |

Upstream sources (pinned in `scripts/r1/fetch_upstream_assets.py`; fetched copies live in the
git-ignored `third_party_assets/` with a `MANIFEST.json`):

| Source | Commit | License |
|---|---|---|
| https://github.com/unitreerobotics/unitree_ros | `ccfc6fd8430a17ba3dacef9a1e2faf64ff3b0aee` | BSD-3-Clause |
| https://github.com/unitreerobotics/unitree_rl_mjlab | `1425b15f73bd4095f0df53709d7c389c3eb9e790` | BSD-3-Clause |
| https://github.com/unitreerobotics/unitree_mujoco | `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d` | BSD-3-Clause (reference only; nothing copied) |

Transformations applied to the URDF: links renamed (`pelvis_link`→`pelvis`,
`waist_yaw_link`→`torso_link`); `head_pitch_joint`/`head_yaw_joint` made fixed; Dex3-1
subtrees attached as fixed joints at `wrist_roll_link + [0.080, 0, 0]` (VERIFY on hardware)
with finger joints fixed at the open pose; wrist collision mesh (contains the stock fist)
replaced by a forearm cylinder; mesh paths made relative (`meshes/<file>`).
