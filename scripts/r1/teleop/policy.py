#!/usr/bin/env python3
"""The exported teleop policy (``export_teleop_onnx.py``) on onnxruntime.

One call maps the flat 1047-float observation (tokenizer terms, then the actor's history terms;
see the ``.json`` written next to the ONNX) to the raw 24-DOF action in Isaac Lab joint order.
The runtime turns it into joint targets as the training env does:
``default_joint_pos + action_scale * clip(action, +-action_clip)``.

    python scripts/r1/teleop/policy.py --check <export>.onnx    # gate G0: ONNX vs training forward
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


class TeleopPolicy:
    def __init__(self, onnx_path: str | Path, device: str = "cpu", threads: int = 4):
        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        self.meta = json.loads(onnx_path.with_suffix(".json").read_text())
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda"):
            if hasattr(ort, "preload_dlls"):  # CUDA / cuDNN from the nvidia-* pip wheels
                ort.preload_dlls()
            index = int(device.split(":")[1]) if ":" in device else 0
            providers.insert(0, ("CUDAExecutionProvider", {"device_id": index}))
        self.session = ort.InferenceSession(str(onnx_path), options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = int(self.meta["input"]["size"])
        assert self.session.get_inputs()[0].shape[-1] == self.input_size

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Raw action (24,), Isaac Lab joint order. ``obs``: (1047,) float."""
        x = np.asarray(obs, np.float32).reshape(1, self.input_size)
        return self.session.run(None, {self.input_name: x})[0][0]


def check(onnx_path: Path, device: str = "cpu") -> float:
    """Max |ONNX - training forward| on the random observations saved by the exporter."""
    policy = TeleopPolicy(onnx_path, device=device)
    data = np.load(str(onnx_path).removesuffix(".onnx") + "_check.npz")
    errors, times = [], []
    for x, y in zip(data["onnx_input"], data["actions"]):
        t = time.perf_counter()
        a = policy(x)
        times.append(time.perf_counter() - t)
        errors.append(np.abs(a - y).max())
    err = float(np.max(errors))
    print(
        f"G0 ONNX vs training forward: max |diff| {err:.2e} over {len(errors)} observations; "
        f"inference {1e3 * np.median(times):.2f} ms median ({device})"
    )
    return err


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", type=Path, required=True, help="exported .onnx")
    ap.add_argument("--device", default="cpu", help="cpu | cuda:<i>")
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args()
    err = check(args.check, args.device)
    raise SystemExit(0 if err < args.tol else 1)


if __name__ == "__main__":
    main()
