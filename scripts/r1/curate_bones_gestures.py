#!/usr/bin/env python3
"""Select and extract S2: real mocap of the demo's gestures (wave, point, beckon, greet, clap).

The Quest demo has to wave, point and beckon well; BONES-SEED has real captures of exactly these
(standing and walking), far better references than text-to-motion samples. S2 takes every
standing wave / point / beckon clip and a spread of greeting and clapping clips, *including the
mirrored captures* (BONES-SEED's ``is_mirror``), so left- and right-handed versions are
balanced, and excludes what S0/S1 already use. Filters as S1 (no jumping, running, sitting,
kneeling, floor work; 1-20 s).

    python scripts/r1/curate_bones_gestures.py --dest data/bones_seed/s2_source
    python gear_sonic/data_process/transfer_g1_motion_lib_to_r1.py \\
        --input data/bones_seed/s2_source/g1/csv --output data/motion_lib_r1/S2 \\
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

# group: (description regex, max clips; None = all)
GROUPS = {
    "wave": (r"\bwav(?:e|es|ing)\b", None),
    "point": (r"\bpoint(?:s|ing)?\b", None),
    "beckon": (r"beckon|come here|come over", None),
    "greet": (r"greet|hello|goodbye|\bbye\b|salut", 250),
    "clap": (r"\bclap", 120),
}
UNWANTED = re.compile(
    r"\b(?:jump|run|jog|sit|kneel|crouch|squat|lie|lying|injur|fall|crawl|floor|ground|danc|drunk"
    r"|zombie)\w*",
    re.I,
)


def group_for(row: dict[str, str]) -> str | None:
    desc = row["content_short_description"]
    if row["is_neutral"] != "1.0" or not row["content_body_position"].startswith("standing"):
        return None
    if not 120 <= int(float(row["move_duration_frames"])) <= 2400 or UNWANTED.search(desc):
        return None
    return next((g for g, (rx, _) in GROUPS.items() if re.search(rx, desc, re.I)), None)


def select(metadata: Path, exclude: set[str]) -> dict[str, list[dict[str, str]]]:
    by_group: dict[str, dict[str, list]] = {g: {} for g in GROUPS}
    with metadata.open(newline="") as stream:
        for row in csv.DictReader(stream):
            group = group_for(row)
            if group is not None and row["move_g1_path"] not in exclude:
                by_group[group].setdefault(row["content_short_description"], []).append(row)
    selected = {}
    for group, (_, cap) in GROUPS.items():
        by_desc = by_group[group]
        for rows in by_desc.values():
            rows.sort(key=lambda r: r["move_g1_path"])
        chosen = []
        limit = cap if cap is not None else sum(len(v) for v in by_desc.values())
        while len(chosen) < limit and any(by_desc.values()):  # round-robin over descriptions
            for desc in sorted(by_desc):
                if by_desc[desc] and len(chosen) < limit:
                    chosen.append(by_desc[desc].pop(0))
        selected[group] = chosen
    return selected


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--metadata", type=Path, default=Path("data/bones_seed/metadata/seed_metadata_v004.csv")
    )
    ap.add_argument("--archive", type=Path, default=Path("data/bones_seed/g1.tar.gz"))
    ap.add_argument("--dest", type=Path, default=Path("data/bones_seed/s2_source"))
    ap.add_argument(
        "--exclude", type=Path, nargs="*",
        default=[Path("data/bones_seed/s0_source_curated/S0_selection.json"),
                 Path("data/bones_seed/s1_source/S1_selection.json")],
    )  # fmt: skip
    ap.add_argument("--select-only", action="store_true")
    args = ap.parse_args()
    exclude = {
        row["move_g1_path"]
        for manifest in args.exclude
        for rows in json.loads(manifest.read_text()).values()
        for row in rows
    }
    selected = select(args.metadata, exclude)
    args.dest.mkdir(parents=True, exist_ok=True)
    manifest = args.dest / "S2_selection.json"
    manifest.write_text(json.dumps(selected, indent=2) + "\n")
    print({g: len(rows) for g, rows in selected.items()}, flush=True)
    if not args.select_only:
        extract(args.archive, args.dest, selected)
    print(f"Selection manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
