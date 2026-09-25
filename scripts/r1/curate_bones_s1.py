#!/usr/bin/env python3
"""Select and extract S1: standing, arm-rich BONES-SEED clips that widen the teleop policy's hand workspace.

At deployment the operator's hands can go anywhere; S0 has only ~150 arm-rich clips, and upper-body
grafting (``upper_body_augment_prefixes: ["planner_"]``) draws its arms from the mocap clips. S1 adds
standing gestures, in-place manipulation and reaching (no bending: the R1 has no waist pitch),
spread across descriptions like S0 and disjoint from S0's selection.

    python scripts/r1/curate_bones_s1.py --dest data/bones_seed/s1_source
    python gear_sonic/data_process/transfer_g1_motion_lib_to_r1.py \
        --input data/bones_seed/s1_source/g1/csv --output data/motion_lib_r1/S1 \
        --fps 30 --fps_source 120 --num_workers 8 --max-clamp-frac 0.05 --max-vel-frac 0.05
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curate_bones_s0 import extract  # noqa: E402

GROUPS = (
    ("gestures", 400),
    ("manipulation", 350),
    ("reach", 250),
)
GESTURE_CATEGORIES = ("Gestures", "Communication", "Looking and Pointing")
MANIPULATION_CATEGORIES = ("Object Manipulation", "Object Interaction", "Household", "Consuming")
# S0's filter plus bending and floor work, which the R1 cannot reproduce without waist pitch.
UNWANTED = re.compile(
    r"\b(?:jump|run|sit|kneel|crouch|squat|lie|lying|injur|fall|crawl|bend|floor|ground|danc)\w*"
    r"|\bpick(?:s|ing)? up\b",
    re.I,
)


def group_for(row: dict[str, str]) -> str | None:
    desc = row["content_short_description"]
    if row["is_mirror"] != "False" or row["is_neutral"] != "1.0":
        return None
    if row["content_body_position"] != "standing":
        return None
    frames = int(float(row["move_duration_frames"]))
    if not 120 <= frames <= 2400 or UNWANTED.search(desc):
        return None
    if re.search(r"\breach", desc, re.I):
        return "reach"
    if row["category"] in GESTURE_CATEGORIES:
        return "gestures"
    if row["category"] in MANIPULATION_CATEGORIES and row["content_vertical_move"] == "0":
        return "manipulation"
    return None


def select(metadata: Path, exclude: set[str]) -> dict[str, list[dict[str, str]]]:
    candidates = {name: {} for name, _ in GROUPS}
    with metadata.open(newline="") as stream:
        for row in csv.DictReader(stream):
            group = group_for(row)
            if group is not None and row["move_g1_path"] not in exclude:
                candidates[group].setdefault(row["content_short_description"], []).append(row)

    selected = {}
    for group, count in GROUPS:
        by_desc = candidates[group]
        for rows in by_desc.values():
            rows.sort(key=lambda row: row["move_g1_path"])
        chosen = []
        # Round-robin over descriptions so no single action or actor fills the quota.
        while len(chosen) < count and any(by_desc.values()):
            for desc in sorted(by_desc):
                if by_desc[desc] and len(chosen) < count:
                    chosen.append(by_desc[desc].pop(0))
        if len(chosen) < count:
            raise ValueError(f"Only {len(chosen)} eligible motions for {group}; need {count}")
        selected[group] = chosen
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/bones_seed/metadata/seed_metadata_v004.csv")
    )
    parser.add_argument("--archive", type=Path, default=Path("data/bones_seed/g1.tar.gz"))
    parser.add_argument("--dest", type=Path, default=Path("data/bones_seed/s1_source"))
    parser.add_argument(
        "--exclude",
        type=Path,
        nargs="*",
        default=[Path("data/bones_seed/s0_source_curated/S0_selection.json")],
        help="selection manifests whose clips S1 must not repeat",
    )
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    exclude = {
        row["move_g1_path"]
        for manifest in args.exclude
        for rows in json.loads(manifest.read_text()).values()
        for row in rows
    }
    selected = select(args.metadata, exclude)
    args.dest.mkdir(parents=True, exist_ok=True)
    manifest = args.dest / "S1_selection.json"
    manifest.write_text(json.dumps(selected, indent=2) + "\n")
    print({group: len(rows) for group, rows in selected.items()}, flush=True)
    if not args.select_only:
        extract(args.archive, args.dest, selected)
    print(f"Selection manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
