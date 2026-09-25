#!/usr/bin/env python3
"""Generate gesture and locomotion clips with Kimodo-G1 (text-to-motion) for R1 training.

Kimodo (NVIDIA, arXiv 2603.15546; https://github.com/nv-tlabs/kimodo) is a kinematic motion
diffusion model trained on 700 h of mocap. Its Unitree-G1 variant writes MuJoCo qpos directly
(root xyz, root quat wxyz, 29 G1 joints, 30 fps), which the joint-name G1 -> R1 transfer takes
as is (``transfer_g1_motion_lib_to_r1.py --csv-format qpos``, contact fix on). Kimodo's own
post-processing is off for G1 (its authors: it does not work well for that model), so foot
cleanup happens in our transfer.

The program below covers what the Quest demo needs done well: waving, pointing, beckoning,
reaching, other arm gestures, idle variety, and walking with and without gestures. Each entry
is generated ``--samples`` times per seed; clips are named ``kimodo_<entry>_s<seed>_<i>`` so the
evaluation can report them as their own group. Kimodo samples vary in quality (BruteForce kept
~1 in 9 "wave" samples), so ``kimodo_quality.py`` filters the transferred clips afterwards.

Text embeddings. Kimodo conditions on LLM2Vec (Llama-3-8B) embeddings. With the versions Kimodo
pins (transformers 5.1), the bidirectional Llama's activations overflow at layer 3 and every
prompt encodes to an all-zero vector, so all prompts give the same unconditional motion (checked
2026-09-25: per-layer mean |h| jumps to 4e30, the final RMSNorm returns 0). The embeddings are
therefore computed by the original ``llm2vec`` package (transformers 4.44, its supported range)
in the ``llm2vec`` env and passed in:

    python scripts/r1/generate_kimodo_motions.py --list-prompts data/kimodo_g1/prompts.json
    python scripts/r1/kimodo_text_embeddings.py data/kimodo_g1/prompts.json   # llm2vec env
    python scripts/r1/generate_kimodo_motions.py --output data/kimodo_g1/K1 \\
        --text-embeddings data/kimodo_g1/prompts.npz --samples 4 --seeds 0 1   # kimodo env
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

# (name, [prompts], [durations s]); several prompts are generated as one continuous clip
PROGRAM = [
    # waving
    ("wave_right", ["A person stands still and waves hello with their right hand."], [5.0]),
    ("wave_left", ["A person stands still and waves hello with their left hand."], [5.0]),
    ("wave_both", ["A person stands still and waves both hands above their head."], [5.0]),
    ("wave_big", ["A person waves enthusiastically at someone far away with the right arm raised high."], [5.0]),
    # pointing
    ("point_forward", ["A person points straight ahead with their right index finger, then lowers the arm."], [5.0]),  # noqa: E501
    ("point_left", ["A person points to the left with their left arm, then lowers it."], [5.0]),
    ("point_right", ["A person points to the right with their right arm, then lowers it."], [5.0]),
    ("point_up", ["A person points up at the sky with their right hand."], [5.0]),
    ("point_down", ["A person points down at the ground in front of them."], [5.0]),
    # beckoning and signalling
    ("beckon", ["A person beckons someone to come closer with their right hand."], [5.0]),
    ("stop_sign", ["A person holds up their right palm to signal stop."], [4.0]),
    ("thumbs_up", ["A person raises their right hand in a thumbs up gesture."], [4.0]),
    ("salute", ["A person gives a military salute with their right hand."], [4.0]),
    ("high_five", ["A person raises their right hand to give a high five."], [4.0]),
    # reaching
    ("reach_forward", ["A person reaches forward with both hands and brings them back."], [5.0]),
    ("reach_up", ["A person reaches up high with their right hand and lowers it."], [5.0]),
    ("reach_side", ["A person reaches out to the side with their left hand and returns."], [5.0]),
    ("reach_cross", ["A person reaches across their body with their right hand to the left side."], [5.0]),
    ("give_object", ["A person holds out their right hand as if handing an object to someone in front."], [5.0]),
    # other arm gestures
    ("clap", ["A person claps their hands several times."], [4.0]),
    ("shrug", ["A person shrugs their shoulders with palms up."], [4.0]),
    ("arms_up", ["A person raises both arms above their head and lowers them."], [5.0]),
    ("arms_out", ["A person stretches both arms out to the sides and brings them down."], [5.0]),
    ("hands_hips", ["A person puts both hands on their hips."], [4.0]),
    ("cross_arms", ["A person crosses their arms in front of their chest."], [4.0]),
    ("present", ["A person presents something with an open right hand, gesturing to the side."], [5.0]),
    ("explain", ["A person talks and gestures with both hands while explaining something."], [6.0]),
    ("look_around", ["A person stands and turns their upper body to look left and then right."], [6.0]),
    # idle
    ("idle", ["A person stands naturally, relaxed, shifting their weight slightly."], [6.0]),
    ("idle_arms", ["A person stands and lets their arms hang loosely, occasionally moving them."], [6.0]),
    # locomotion
    ("walk_forward", ["A person walks forward slowly and stops."], [6.0]),
    ("walk_backward", ["A person walks backward slowly and stops."], [5.0]),
    ("sidestep_left", ["A person takes a few steps sideways to the left."], [5.0]),
    ("sidestep_right", ["A person takes a few steps sideways to the right."], [5.0]),
    ("turn_around", ["A person turns around in place."], [5.0]),
    ("walk_turn", ["A person walks forward and turns left."], [6.0]),
    # locomotion + gestures
    ("walk_wave", ["A person walks forward while waving with their right hand."], [6.0]),
    ("walk_point", ["A person walks forward and points ahead with the right arm."], [6.0]),
    ("walk_then_wave", ["A person walks forward.", "A person stops and waves with their right hand."], [3.0, 4.0]),
    ("wave_then_walk", ["A person waves with their left hand.", "A person walks forward."], [3.0, 4.0]),
    ("walk_then_point", ["A person walks forward.", "A person stops and points to the right."], [3.0, 4.0]),
    ("walk_then_beckon", ["A person walks forward slowly.", "A person stops and beckons with their right hand."], [3.0, 4.0]),  # noqa: E501
]  # fmt: skip


class PrecomputedText:
    """Stands in for Kimodo's text encoder: (N, 1, 4096) embeddings looked up by prompt."""

    def __init__(self, path: Path, device: str):
        import numpy as np
        import torch

        z = np.load(path)
        self.table = {str(t): torch.from_numpy(e).float() for t, e in zip(z["texts"], z["emb"])}
        self.device = device

    def __call__(self, texts):
        import torch

        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        missing = [t for t in texts if t not in self.table]
        if missing:
            raise KeyError(f"no precomputed embedding for {missing}")
        feat = torch.stack([self.table[t] for t in texts])[:, None].to(self.device)
        lengths = [1] * len(texts)
        return (feat[0], 1) if single else (feat, lengths)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--list-prompts", type=Path, default=None, help="write every prompt (json) and exit"
    )
    ap.add_argument(
        "--text-embeddings", type=Path, default=None, help="npz from kimodo_text_embeddings.py"
    )
    ap.add_argument("--model", default="Kimodo-G1-RP-v1")
    ap.add_argument("--samples", type=int, default=4, help="samples per entry and seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=100, help="denoising steps")
    ap.add_argument(
        "--cfg", type=float, nargs=2, default=[2.0, 2.0],
        help="separated classifier-free guidance [text, constraint] (Kimodo's G1 examples: 2 2)",
    )  # fmt: skip
    ap.add_argument("--only", nargs="*", default=None, help="entry names to generate")
    args = ap.parse_args()
    if args.list_prompts:
        prompts = sorted({t for _, texts, _ in PROGRAM for t in texts})
        args.list_prompts.parent.mkdir(parents=True, exist_ok=True)
        args.list_prompts.write_text(json.dumps(prompts, indent=1))
        print(f"{len(prompts)} prompts -> {args.list_prompts}")
        return
    assert args.output is not None, "--output is required"

    from kimodo import load_model
    from kimodo.exports.mujoco import MujocoQposConverter
    from kimodo.tools import seed_everything
    import torch

    os.environ.setdefault("TEXT_ENCODER_DEVICE", "cpu")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, resolved = load_model(
        args.model, device=device, default_family="Kimodo", return_resolved_name=True
    )
    assert "g1" in resolved, f"{resolved}: a G1 model is needed for MuJoCo qpos output"
    if args.text_embeddings:
        model.text_encoder = PrecomputedText(args.text_embeddings, device)
    converter = MujocoQposConverter(model.skeleton)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, prompts, durations in PROGRAM:
        if args.only and name not in args.only:
            continue
        num_frames = [int(round(d * model.fps)) for d in durations]
        for seed in args.seeds:
            seed_everything(seed)
            t0 = time.time()
            out = model(
                prompts, num_frames, constraint_lst=[], num_denoising_steps=args.steps,
                num_samples=args.samples, multi_prompt=True, num_transition_frames=5,
                post_processing=False, return_numpy=True,
                cfg_type="separated", cfg_weight=list(args.cfg),
            )  # fmt: skip
            qpos = converter.dict_to_qpos(out, device)  # (samples, T, 36)
            qpos = qpos.cpu().numpy() if hasattr(qpos, "cpu") else qpos
            for i, q in enumerate(qpos):
                stem = f"kimodo_{name}_s{seed}_{i:02d}"
                path = args.output / f"{stem}.csv"
                with path.open("w") as f:
                    for row in q:
                        f.write(",".join(f"{x:.6f}" for x in row) + "\n")
                manifest.append(
                    {"clip": stem, "entry": name, "prompts": prompts, "durations": durations,
                     "seed": seed, "sample": i, "frames": int(len(q)), "fps": float(model.fps)}
                )  # fmt: skip
            print(f"{name} seed {seed}: {len(qpos)} samples, {time.time() - t0:.1f} s", flush=True)
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest)} clips in {args.output}")


if __name__ == "__main__":
    main()
