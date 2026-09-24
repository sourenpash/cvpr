#!/usr/bin/env python3
"""Checkpoint surgery: initialise an R1 (24-DOF) SONIC policy from G1 (29-DOF) weights.

Every tensor whose shape already matches is copied verbatim (token space / FSQ, all hidden
layers, the teleop encoder, most of the critic). Tensors whose
shape depends on the embodiment are *gathered* along the mismatched axes using index maps
derived from the two observation layouts (``layout.json`` written by
``++dump_layout_dir=...``):

* action rows / std          : joint-name map, Isaac Lab DOF order (all 24 R1 joints exist on G1)
* policy / critic obs columns: per-term maps (joint-indexed terms by name, history-major)
* decoder input columns      : identity for ``token_flattened`` + policy-obs map
* encoder input columns      : time-major concat of the encoder's input terms
* kinematic decoder rows    : time-major concat of its output terms
* tokenizer-wide vectors     : term-order concat (same rule)

Nothing is invented: because R1's joints are a subset of G1's, every target index has a
source index. Tensors that cannot be mapped keep the template's random initialisation and
are listed in the report. Output is a checkpoint in ``ModelSaveCallback`` format
(``policy_state_dict`` + ``value_state_dict``), loadable with ``+checkpoint=<out>``.

Usage (GPU box, after dumping both layouts):
    python scripts/r1/surgery_g1_to_r1_checkpoint.py \
        --src sonic_v1_1/last.pt --src-layout layouts/g1_v1_1/layout.json \
        --tgt layouts/r1_dex3/template.pt --tgt-layout layouts/r1_dex3/layout.json \
        --out r1_init/last.pt [--dry-run]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import re

import numpy as np

# Observation terms whose per-step vector is indexed by Isaac Lab DOF (possibly with history).
DOF_TERMS = {
    "joint_pos",
    "joint_vel",
    "actions",
    "last_actions",
    "joint_pos_wo_hand",
    "joint_vel_wo_hand",
    "actions_wo_hand",
    "joint_pos_rel",
    "joint_vel_rel",
}
# [pos: T x dof][vel: T x dof], Isaac Lab DOF order. The nonflat observation
# reshapes this same vector to (T, 2*dof) without changing its flat order.
DOF_MULTI_FUTURE_POSVEL_TERMS = {"command_multi_future", "command_multi_future_nonflat"}
# (T, len(joints_idx)) frame-major; joints_idx are Isaac Lab DOF indices.
WRIST_TERMS = {"joint_pos_multi_future_wrist_for_smpl", "joint_pos_multi_future_wrist_for_soma"}


@dataclass
class IndexMap:
    """tgt index -> src index (all entries >= 0 for a pure gather)."""

    name: str
    tgt_to_src: np.ndarray
    source_len: int | None = None

    @property
    def src_len(self) -> int:
        if self.source_len is not None:
            return self.source_len
        return int(self.tgt_to_src.max()) + 1 if len(self.tgt_to_src) else 0

    @property
    def tgt_len(self) -> int:
        return len(self.tgt_to_src)


@dataclass
class Layout:
    raw: dict
    joint_names: list[str]
    num_actions: int

    @classmethod
    def load(cls, path: str | Path) -> "Layout":
        raw = json.loads(Path(path).read_text())
        return cls(
            raw=raw,
            joint_names=list(raw["joint_names_isaaclab"]),
            num_actions=int(raw["num_actions"]),
        )

    def group_terms(self, group: str) -> list[dict]:
        return list(self.raw["groups"].get(group, []))

    def tokenizer_dims(self, term: str) -> list[int]:
        return list(self.raw["tokenizer_term_dims"][term])


class MapBuilder:
    """Build all index maps between a source (G1) and target (R1) layout."""

    def __init__(self, src: Layout, tgt: Layout):
        self.src, self.tgt = src, tgt
        missing = [j for j in tgt.joint_names if j not in src.joint_names]
        if missing:
            raise ValueError(
                f"target joints missing from source (surgery would need re-init): {missing}"
            )
        self.joint_map = np.array(
            [src.joint_names.index(j) for j in tgt.joint_names], dtype=np.int64
        )
        self.maps: dict[str, IndexMap] = {}
        self._build()

    # ---- primitive maps -------------------------------------------------------------
    def _dof_hist_map(self, src_dim: int, tgt_dim: int, src_hist: int, tgt_hist: int) -> np.ndarray:
        n_src, n_tgt = len(self.src.joint_names), len(self.tgt.joint_names)
        if src_hist != tgt_hist:
            raise ValueError("history length differs between layouts")
        k_src, k_tgt = src_dim // (src_hist * n_src), tgt_dim // (tgt_hist * n_tgt)
        if (
            k_src * src_hist * n_src != src_dim
            or k_tgt * tgt_hist * n_tgt != tgt_dim
            or k_src != k_tgt
        ):
            raise ValueError(
                f"cannot interpret DOF term dims {src_dim}->{tgt_dim} with {n_src}->{n_tgt} joints"
            )
        per_step_src = k_src * n_src
        out = []
        for h in range(tgt_hist):
            for k in range(k_tgt):
                out.extend(h * per_step_src + k * n_src + self.joint_map)
        assert len(out) == tgt_dim
        return np.array(out, dtype=np.int64)

    def _dof_mf_posvel_map(self, src_dim: int, tgt_dim: int) -> np.ndarray:
        n_src, n_tgt = len(self.src.joint_names), len(self.tgt.joint_names)
        T = tgt_dim // (2 * n_tgt)
        if 2 * T * n_tgt != tgt_dim or 2 * T * n_src != src_dim:
            raise ValueError(f"cannot interpret command_multi_future dims {src_dim}->{tgt_dim}")
        out = []
        for block in range(2):  # pos, vel
            for t in range(T):
                out.extend(block * T * n_src + t * n_src + self.joint_map)
        return np.array(out, dtype=np.int64)

    def _wrist_map(self, src_dims: list[int], tgt_dims: list[int]) -> np.ndarray:
        src_idx = list(self.src.raw.get("wrist_for_smpl_joints_idx", []))
        tgt_idx = list(self.tgt.raw.get("wrist_for_smpl_joints_idx", []))
        T_s, W_s = (src_dims + [1])[:2] if len(src_dims) >= 2 else (1, src_dims[0])
        T_t, W_t = (tgt_dims + [1])[:2] if len(tgt_dims) >= 2 else (1, tgt_dims[0])
        if T_s != T_t or len(src_idx) != W_s or len(tgt_idx) != W_t:
            raise ValueError(
                f"wrist term dims/joints_idx inconsistent: {src_dims} {src_idx} -> {tgt_dims} {tgt_idx}"
            )
        src_names = [self.src.joint_names[i] for i in src_idx]
        tgt_names = [self.tgt.joint_names[i] for i in tgt_idx]
        col = [src_names.index(n) for n in tgt_names]
        out = []
        for t in range(T_t):
            out.extend(t * W_s + np.array(col, dtype=np.int64))
        return np.array(out, dtype=np.int64)

    def _term_map(self, name: str, src_dims, tgt_dims, src_hist=1, tgt_hist=1) -> np.ndarray:
        s, t = int(np.prod(src_dims)), int(np.prod(tgt_dims))
        if name in DOF_TERMS:
            return self._dof_hist_map(s, t, src_hist, tgt_hist)
        if name in DOF_MULTI_FUTURE_POSVEL_TERMS:
            return self._dof_mf_posvel_map(s, t)
        if name in WRIST_TERMS:
            return self._wrist_map(list(src_dims), list(tgt_dims))
        if s != t:
            raise ValueError(f"term {name!r} has no mapping rule but dims differ ({s} -> {t})")
        return np.arange(t, dtype=np.int64)

    # ---- composite maps -------------------------------------------------------------
    def _group_map(self, group: str) -> np.ndarray | None:
        src_terms = {d["name"]: d for d in self.src.group_terms(group)}
        tgt_terms = self.tgt.group_terms(group)
        if not tgt_terms:
            return None
        out, src_off = [], 0
        for d in tgt_terms:
            s = src_terms.get(d["name"])
            if s is None:
                raise ValueError(f"group {group}: term {d['name']} missing in source layout")
            m = self._term_map(
                d["name"], s["dims"], d["dims"], s.get("history", 1), d.get("history", 1)
            )
            out.extend(src_off + m)
            src_off += int(np.prod(s["dims"]))
        return np.array(out, dtype=np.int64)

    def _temporal_concat_map(self, module: str, inputs: list[str]) -> np.ndarray:
        """Map per-frame concatenated encoder inputs or decoder outputs."""
        src_pf, tgt_pf, per_frame_maps, T = [], [], [], None
        for name in inputs:
            sd, td = self.src.tokenizer_dims(name), self.tgt.tokenizer_dims(name)
            if len(td) >= 2:
                if sd[0] != td[0]:
                    raise ValueError(f"{module}: input {name} has different temporal dimensions")
                T_i, ds, dt = td[0], int(np.prod(sd[1:])), int(np.prod(td[1:]))
            else:  # 1-D input broadcast to every frame is not expected; treat as T=1
                T_i, ds, dt = 1, int(np.prod(sd)), int(np.prod(td))
            T = T_i if T is None else T
            if T_i != T:
                raise ValueError(f"{module}: inputs have different temporal dims")
            m_full = self._term_map(name, sd, td)
            m_pf = m_full[:dt]
            for frame in range(T_i):
                if not np.array_equal(m_full[frame * dt : (frame + 1) * dt], frame * ds + m_pf):
                    raise ValueError(f"{module}: input {name} is not frame-local")
            src_pf.append(ds)
            tgt_pf.append(dt)
            per_frame_maps.append(m_pf)
        src_frame, out = sum(src_pf), []
        for t in range(T):
            off = 0
            for ds, m_pf in zip(src_pf, per_frame_maps):
                out.extend(t * src_frame + off + m_pf)
                off += ds
        return np.array(out, dtype=np.int64)

    def _build(self):
        self.maps["action"] = IndexMap("action", self.joint_map.copy(), len(self.src.joint_names))
        for group in ("policy", "critic"):
            m = self._group_map(group)
            if m is not None:
                src_size = sum(int(np.prod(d["dims"])) for d in self.src.group_terms(group))
                self.maps[f"group:{group}"] = IndexMap(f"group:{group}", m, src_size)
        tok_dim = int(self.src.raw.get("token_dim", 0)) * int(self.src.raw.get("max_num_tokens", 0))
        pol = self.maps.get("group:policy")
        if pol is not None and tok_dim > 0:
            self.maps["decoder_input"] = IndexMap(
                "decoder_input",
                np.concatenate([np.arange(tok_dim), tok_dim + pol.tgt_to_src]),
                tok_dim + pol.src_len,
            )
        for enc, inputs in self.tgt.raw.get("encoder_inputs", {}).items():
            if inputs:
                src_size = sum(int(np.prod(self.src.tokenizer_dims(n))) for n in inputs)
                self.maps[f"encoder:{enc}"] = IndexMap(
                    f"encoder:{enc}",
                    self._temporal_concat_map(f"encoder {enc}", inputs),
                    src_size,
                )
        for dec, outputs in self.tgt.raw.get("decoder_outputs", {}).items():
            if outputs == ["action"]:
                self.maps[f"decoder_output:{dec}"] = IndexMap(
                    f"decoder_output:{dec}", self.joint_map.copy(), len(self.src.joint_names)
                )
            elif outputs:
                src_size = sum(int(np.prod(self.src.tokenizer_dims(n))) for n in outputs)
                self.maps[f"decoder_output:{dec}"] = IndexMap(
                    f"decoder_output:{dec}",
                    self._temporal_concat_map(f"decoder {dec}", outputs),
                    src_size,
                )
        # tokenizer-wide vector (term order)
        order = self.tgt.raw.get("tokenizer_term_order", [])
        if order:
            out, src_off = [], 0
            for name in order:
                sd, td = self.src.tokenizer_dims(name), self.tgt.tokenizer_dims(name)
                out.extend(src_off + self._term_map(name, sd, td))
                src_off += int(np.prod(sd))
            self.maps["tokenizer"] = IndexMap("tokenizer", np.array(out, dtype=np.int64), src_off)


# --------------------------------------------------------------------------------------
# Applying maps to a state dict
# --------------------------------------------------------------------------------------
@dataclass
class SurgeryReport:
    copied: list[str] = field(default_factory=list)
    gathered: dict[str, str] = field(default_factory=dict)  # key -> description of maps used
    kept_init: dict[str, str] = field(default_factory=dict)  # key -> reason
    src_only: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "copied": len(self.copied),
            "gathered": len(self.gathered),
            "kept_template_init": len(self.kept_init),
            "source_only_ignored": len(self.src_only),
            "gathered_detail": self.gathered,
            "kept_template_init_detail": self.kept_init,
        }


def _candidates(maps: dict[str, IndexMap], src_len: int, tgt_len: int, key: str) -> list[IndexMap]:
    cands = [m for m in maps.values() if m.src_len == src_len and m.tgt_len == tgt_len]
    if len(cands) <= 1:
        return cands
    # disambiguate by parameter name
    prefs = []
    dec_match = re.search(r"decoders\.(\w+)", key)
    if dec_match and re.search(r"(?:\.|^)(?:module|net)\.\d+\.(?:weight|bias)$", key):
        prefs.append("decoder_output:" + dec_match.group(1))
    if dec_match:
        prefs.append("decoder_input")
    enc_match = re.search(r"encoders\.(\w+)", key)
    if enc_match:
        prefs.append("encoder:" + enc_match.group(1))
    if "std" in key or key.endswith("action") or re.search(r"decoders\..*(output|out)", key):
        prefs.append("action")
    if "value" in key or "critic" in key:
        prefs.append("group:critic")
    for p in prefs:
        for c in cands:
            if c.name == p:
                return [c]
    return cands


def surgery(
    src_sd: dict, tgt_sd: dict, maps: dict[str, IndexMap], report: SurgeryReport, prefix: str = ""
):
    import torch

    out = {}
    for key, tgt_t in tgt_sd.items():
        full = prefix + key
        if key not in src_sd:
            out[key] = tgt_t
            report.kept_init[full] = "not in source"
            continue
        src_t = src_sd[key]
        if not torch.is_tensor(tgt_t) or not torch.is_tensor(src_t):
            out[key] = src_t if type(src_t) is type(tgt_t) else tgt_t
            continue
        if tuple(src_t.shape) == tuple(tgt_t.shape):
            out[key] = src_t.clone()
            report.copied.append(full)
            continue
        if src_t.ndim != tgt_t.ndim or src_t.ndim > 2:
            out[key] = tgt_t
            report.kept_init[full] = (
                f"rank/shape unsupported {tuple(src_t.shape)}->{tuple(tgt_t.shape)}"
            )
            continue
        gathered = src_t
        used = []
        ok = True
        for axis in range(tgt_t.ndim):
            s_len, t_len = src_t.shape[axis], tgt_t.shape[axis]
            if s_len == t_len:
                continue
            cands = _candidates(maps, s_len, t_len, full)
            if len(cands) != 1:
                ok = False
                reason = "no" if not cands else f"ambiguous {[c.name for c in cands]}"
                report.kept_init[full] = f"{reason} map for axis {axis} ({s_len}->{t_len})"
                break
            idx = torch.as_tensor(cands[0].tgt_to_src, device=gathered.device, dtype=torch.long)
            gathered = torch.index_select(gathered, axis, idx)
            used.append(f"axis{axis}={cands[0].name}")
        if ok:
            assert tuple(gathered.shape) == tuple(tgt_t.shape)
            out[key] = gathered.clone()
            report.gathered[full] = ", ".join(used)
        else:
            out[key] = tgt_t
    for key in src_sd:
        if key not in tgt_sd:
            report.src_only.append(prefix + key)
    return out


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--src", required=True, help="G1 checkpoint (.pt with policy_state_dict/value_state_dict)"
    )
    ap.add_argument("--src-layout", required=True)
    ap.add_argument(
        "--tgt", required=True, help="R1 template checkpoint (template.pt from ++dump_layout_dir)"
    )
    ap.add_argument("--tgt-layout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_l, tgt_l = Layout.load(args.src_layout), Layout.load(args.tgt_layout)
    builder = MapBuilder(src_l, tgt_l)
    print("index maps:")
    for m in builder.maps.values():
        print(f"  {m.name:20s} {m.src_len:6d} -> {m.tgt_len:6d}")

    src_ck = torch.load(args.src, map_location="cpu", weights_only=False)
    tgt_ck = torch.load(args.tgt, map_location="cpu", weights_only=False)
    report = SurgeryReport()
    out_ck = {
        k: v
        for k, v in tgt_ck.items()
        if k not in ("optimizer_state_dict", "lr_scheduler_state_dict")
    }
    for sd_key in ("policy_state_dict", "value_state_dict", "disc_state_dict"):
        if tgt_ck.get(sd_key) is None:
            continue
        if src_ck.get(sd_key) is None:
            report.kept_init[sd_key] = "state dict absent in source"
            continue
        out_ck[sd_key] = surgery(
            src_ck[sd_key], tgt_ck[sd_key], builder.maps, report, prefix=sd_key + "."
        )
    out_ck["optimizer_state_dict"] = None
    out_ck["lr_scheduler_state_dict"] = None
    # PPOTrainer.load_checkpoint prints this field even for a weights-only warm start.
    from types import SimpleNamespace

    out_ck["state"] = SimpleNamespace(global_step=0)
    out_ck["surgery"] = {
        "src": args.src,
        "tgt": args.tgt,
        "maps": {m.name: [m.src_len, m.tgt_len] for m in builder.maps.values()},
    }

    summary = report.summary()
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("_detail")}, indent=2))
    for k, v in summary["gathered_detail"].items():
        print(f"  gathered  {k}: {v}")
    for k, v in summary["kept_template_init_detail"].items():
        print(f"  kept-init {k}: {v}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".surgery_report.json").write_text(json.dumps(summary, indent=2))
    if args.dry_run:
        print("dry run: report written, checkpoint not written")
        return
    torch.save(out_ck, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
