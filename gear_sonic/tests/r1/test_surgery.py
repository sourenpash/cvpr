"""Checkpoint-surgery tests on small synthetic layouts (G1-like 5 joints -> R1-like 3 joints)."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "surgery", REPO / "scripts/r1/surgery_g1_to_r1_checkpoint.py"
)
surgery = importlib.util.module_from_spec(_spec)
sys.modules["surgery"] = surgery
_spec.loader.exec_module(surgery)

SRC_JOINTS = ["a", "b", "c", "d", "e"]  # G1-like, Isaac Lab order
TGT_JOINTS = ["a", "c", "e"]  # R1-like subset
H = 2  # history
T = 3  # future frames
TOKEN_DIM, NUM_TOKENS = 4, 2


def make_layout(joints, wrist_idx, wrist_w):
    n = len(joints)
    return {
        "robot_type": "x",
        "num_actions": n,
        "joint_names_isaaclab": joints,
        "body_names_isaaclab": [],
        "groups": {
            "policy": [
                {"name": "gravity_dir", "dims": [3 * H], "history": H},
                {"name": "joint_pos", "dims": [n * H], "history": H},
                {"name": "actions", "dims": [n * H], "history": H},
            ],
            "critic": [
                {"name": "command_multi_future", "dims": [2 * T * n], "history": 1},
                {"name": "body_pos", "dims": [6], "history": 1},
                {"name": "joint_vel", "dims": [n * H], "history": H},
            ],
        },
        "tokenizer_term_dims": {
            "smpl_joints": [T, 4],
            "joint_pos_multi_future_wrist_for_smpl": [T, wrist_w],
            "encoder_index": [3],
        },
        "tokenizer_term_order": [
            "encoder_index",
            "smpl_joints",
            "joint_pos_multi_future_wrist_for_smpl",
        ],
        "encoder_inputs": {
            "smpl": ["smpl_joints", "joint_pos_multi_future_wrist_for_smpl"],
            "teleop": ["smpl_joints"],
        },
        "proprioception_features": ["actor_obs"],
        "token_dim": TOKEN_DIM,
        "max_num_tokens": NUM_TOKENS,
        "wrist_for_smpl_joints_idx": wrist_idx,
        "obs_dims": {},
    }


@pytest.fixture(scope="module")
def layouts(tmp_path_factory):
    d = tmp_path_factory.mktemp("layouts")
    import json

    src = make_layout(SRC_JOINTS, wrist_idx=[1, 2, 3, 4], wrist_w=4)  # wrists b,c,d,e
    tgt = make_layout(TGT_JOINTS, wrist_idx=[1, 2], wrist_w=2)  # wrists c,e
    (d / "src.json").write_text(json.dumps(src))
    (d / "tgt.json").write_text(json.dumps(tgt))
    return surgery.Layout.load(d / "src.json"), surgery.Layout.load(d / "tgt.json")


def test_maps(layouts):
    src, tgt = layouts
    b = surgery.MapBuilder(src, tgt)
    assert b.maps["action"].tgt_to_src.tolist() == [0, 2, 4]
    assert b.maps["action"].src_len == len(SRC_JOINTS)
    # policy: gravity(6) | joint_pos hist-major (H x n) | actions (H x n)
    pol = b.maps["group:policy"].tgt_to_src
    n_s, n_t = 5, 3
    expect = list(range(6))
    for term_off in (6, 6 + H * n_s):
        for h in range(H):
            expect += [term_off + h * n_s + j for j in (0, 2, 4)]
    assert pol.tolist() == expect
    assert b.maps["group:policy"].tgt_len == 6 + 2 * H * n_t
    # critic: command_multi_future [pos T x n][vel T x n], body_pos identity, joint_vel hist
    cri = b.maps["group:critic"].tgt_to_src
    expect = []
    for block in range(2):
        for t in range(T):
            expect += [block * T * n_s + t * n_s + j for j in (0, 2, 4)]
    off = 2 * T * n_s
    expect += [off + i for i in range(6)]
    off += 6
    for h in range(H):
        expect += [off + h * n_s + j for j in (0, 2, 4)]
    assert cri.tolist() == expect
    # decoder input: identity token prefix then policy map
    dec = b.maps["decoder_input"].tgt_to_src
    tok = TOKEN_DIM * NUM_TOKENS
    assert dec[:tok].tolist() == list(range(tok)) and (dec[tok:] == tok + pol).all()
    # smpl encoder: time-major [smpl(4) | wrist(4->2)] per frame; wrist cols for c,e = 1,3 of (b,c,d,e)
    enc = b.maps["encoder:smpl"].tgt_to_src
    expect = []
    for t in range(T):
        base = t * (4 + 4)
        expect += [base + i for i in range(4)] + [base + 4 + 1, base + 4 + 3]
    assert enc.tolist() == expect
    # teleop encoder unchanged -> identity
    assert b.maps["encoder:teleop"].tgt_to_src.tolist() == list(range(T * 4))
    # tokenizer-wide vector: encoder_index(3) | smpl(12) | wrist(12->6)
    tokv = b.maps["tokenizer"].tgt_to_src
    expect = list(range(3 + 12)) + [15 + t * 4 + c for t in range(T) for c in (1, 3)]
    assert tokv.tolist() == expect


def test_surgery_gathers_weights(layouts):
    src, tgt = layouts
    b = surgery.MapBuilder(src, tgt)
    n_s, n_t = 5, 3
    pol_s, pol_t = 6 + 2 * H * n_s, 6 + 2 * H * n_t
    tok = TOKEN_DIM * NUM_TOKENS
    g = torch.Generator().manual_seed(0)

    def rnd(*shape):
        return torch.randn(*shape, generator=g)

    src_sd = {
        "backbone.hidden.weight": rnd(8, 8),  # same shape -> copied
        "backbone.decoders.g1_dyn.net.0.weight": rnd(8, tok + pol_s),  # cols gathered
        "backbone.decoders.g1_dyn.net.2.weight": rnd(n_s, 8),  # rows gathered (action)
        "backbone.decoders.g1_dyn.net.2.bias": rnd(n_s),
        "std": torch.full((n_s,), 0.05),
        "backbone.encoders.smpl.net.0.weight": rnd(8, T * 8),
        "running_mean_std.running_mean": rnd(pol_s),
        "only_in_src": rnd(2),
    }
    tgt_sd = {
        "backbone.hidden.weight": torch.zeros(8, 8),
        "backbone.decoders.g1_dyn.net.0.weight": torch.zeros(8, tok + pol_t),
        "backbone.decoders.g1_dyn.net.2.weight": torch.zeros(n_t, 8),
        "backbone.decoders.g1_dyn.net.2.bias": torch.zeros(n_t),
        "std": torch.zeros(n_t),
        "backbone.encoders.smpl.net.0.weight": torch.zeros(8, T * 6),
        "running_mean_std.running_mean": torch.zeros(pol_t),
        "new_module.weight": torch.ones(2, 2),
    }
    rep = surgery.SurgeryReport()
    out = surgery.surgery(src_sd, tgt_sd, b.maps, rep, prefix="policy_state_dict.")

    assert torch.equal(out["backbone.hidden.weight"], src_sd["backbone.hidden.weight"])
    # action rows: joints a,c,e = src rows 0,2,4
    assert torch.equal(
        out["backbone.decoders.g1_dyn.net.2.weight"],
        src_sd["backbone.decoders.g1_dyn.net.2.weight"][[0, 2, 4]],
    )
    assert torch.equal(out["std"], src_sd["std"][[0, 2, 4]])
    # decoder input columns
    dec_map = torch.as_tensor(b.maps["decoder_input"].tgt_to_src)
    assert torch.equal(
        out["backbone.decoders.g1_dyn.net.0.weight"],
        src_sd["backbone.decoders.g1_dyn.net.0.weight"][:, dec_map],
    )
    # smpl encoder columns
    enc_map = torch.as_tensor(b.maps["encoder:smpl"].tgt_to_src)
    assert torch.equal(
        out["backbone.encoders.smpl.net.0.weight"],
        src_sd["backbone.encoders.smpl.net.0.weight"][:, enc_map],
    )
    # normalizer buffer over policy obs
    pol_map = torch.as_tensor(b.maps["group:policy"].tgt_to_src)
    assert torch.equal(
        out["running_mean_std.running_mean"], src_sd["running_mean_std.running_mean"][pol_map]
    )
    # bookkeeping
    assert torch.equal(out["new_module.weight"], tgt_sd["new_module.weight"])
    assert "policy_state_dict.new_module.weight" in rep.kept_init
    assert "policy_state_dict.only_in_src" in rep.src_only
    assert len(rep.copied) == 1 and len(rep.gathered) == 6


def test_nonflat_future_and_kinematic_decoder_follow_source_order(layouts):
    """The nonflat view reshapes a pos-block/vel-block vector without interleaving it."""
    src, tgt = copy.deepcopy(layouts)
    src.raw["tokenizer_term_dims"].update(
        {"command_multi_future_nonflat": [T, 2 * len(SRC_JOINTS)], "anchor": [T, 2]}
    )
    tgt.raw["tokenizer_term_dims"].update(
        {"command_multi_future_nonflat": [T, 2 * len(TGT_JOINTS)], "anchor": [T, 2]}
    )
    for layout in (src, tgt):
        layout.raw["encoder_inputs"]["g1"] = ["command_multi_future_nonflat", "anchor"]
        layout.raw["decoder_outputs"] = {"g1_kin": ["command_multi_future_nonflat", "anchor"]}
    b = surgery.MapBuilder(src, tgt)
    future = b.maps["encoder:g1"].tgt_to_src
    assert np.array_equal(future, b.maps["decoder_output:g1_kin"].tgt_to_src)
    expected = []
    for frame in range(T):
        base = frame * (2 * len(SRC_JOINTS) + 2)
        expected += [base + j for j in (0, 2, 4)]
        expected += [base + len(SRC_JOINTS) + j for j in (0, 2, 4)]
        expected += [base + 2 * len(SRC_JOINTS), base + 2 * len(SRC_JOINTS) + 1]
    assert future.tolist() == expected
    src_sd = {
        "actor_module.decoders.g1_kin.module.8.bias": torch.arange(len(SRC_JOINTS) * 2 * T + 2 * T)
    }
    tgt_sd = {
        "actor_module.decoders.g1_kin.module.8.bias": torch.zeros(len(TGT_JOINTS) * 2 * T + 2 * T)
    }
    report = surgery.SurgeryReport()
    out = surgery.surgery(src_sd, tgt_sd, b.maps, report)
    assert out[next(iter(out))].tolist() == expected
    assert report.kept_init == {}


def test_missing_target_joint_is_rejected(layouts):
    src, tgt = layouts
    bad = surgery.Layout(raw=dict(tgt.raw), joint_names=["a", "zz"], num_actions=2)
    with pytest.raises(ValueError, match="missing from source"):
        surgery.MapBuilder(src, bad)


def test_source_size_includes_unselected_last_joint(layouts):
    src, tgt = copy.deepcopy(layouts)
    tgt.joint_names = ["a", "c"]
    tgt.raw["groups"]["policy"][1]["dims"] = [2 * H]
    tgt.raw["groups"]["policy"][2]["dims"] = [2 * H]
    tgt.raw["groups"]["critic"][0]["dims"] = [2 * T * 2]
    tgt.raw["groups"]["critic"][2]["dims"] = [2 * H]
    tgt.raw["tokenizer_term_dims"]["joint_pos_multi_future_wrist_for_smpl"] = [T, 1]
    tgt.raw["wrist_for_smpl_joints_idx"] = [1]
    b = surgery.MapBuilder(src, tgt)
    assert b.maps["action"].src_len == 5
    assert b.maps["group:policy"].src_len == 6 + 2 * H * 5
    assert b.maps["group:critic"].src_len == 2 * T * 5 + 6 + H * 5


def test_ambiguous_shape_is_kept_init(layouts):
    """Two different maps with identical (src,tgt) lengths must not be applied blindly."""
    src, tgt = layouts
    b = surgery.MapBuilder(src, tgt)
    m = b.maps["action"]
    maps = {"action": m, "other": surgery.IndexMap("other", np.array([4, 2, 0]))}
    rep = surgery.SurgeryReport()
    out = surgery.surgery({"w": torch.randn(5)}, {"w": torch.zeros(3)}, maps, rep)
    assert torch.equal(out["w"], torch.zeros(3)) and "ambiguous" in rep.kept_init["w"]
