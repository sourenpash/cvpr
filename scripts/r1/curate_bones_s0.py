#!/usr/bin/env python3
"""Select and extract a balanced R1 bring-up subset from BONES-SEED's G1 archive.

The archive stays compressed; only metadata-selected, non-mirrored CSVs are extracted.
Selection is deterministic and spreads each group across different motion descriptions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile

GROUPS = (
    ("neutral_stand", 100),
    ("idle_transitions", 100),
    ("basic_walk", 150),
    ("standing_arms", 100),
    ("reach", 50),
)
UNWANTED_POSES = re.compile(r"\b(?:jump|run|sit|kneel|crouch|lie|lying|injur|fall|crawl)\w*", re.I)


def group_for(row: dict[str, str]) -> str | None:
    desc = row["content_short_description"].lower()
    category = row["category"]
    if row["is_mirror"] != "False" or row["is_neutral"] != "1.0":
        return None
    frames = int(float(row["move_duration_frames"]))
    if not 120 <= frames <= 2400 or UNWANTED_POSES.search(desc):
        return None
    if desc == "neutral position":
        return "neutral_stand"
    if category == "Basic Locomotion Neutral" and re.search(r"idle|stand", row["move_name"], re.I):
        return "idle_transitions"
    if (
        category in ("Basic Locomotion Neutral", "Baseline")
        and re.search(r"walk|stroll", desc, re.I)
        and re.search(r"slow|normal pace|stroll|walking forward", desc, re.I)
        and not re.search(r"\b(?:fast|quick)\b", desc, re.I)
    ):
        return "basic_walk"
    if re.search(r"reach(?:ing)? (?:up|down|far)", desc, re.I):
        return "reach"
    if category in ("Gestures", "Communication", "Looking and Pointing") and re.search(
        r"arm|hand|point", desc, re.I
    ):
        return "standing_arms"
    return None


def select(metadata: Path) -> dict[str, list[dict[str, str]]]:
    candidates = {name: {} for name, _ in GROUPS}
    with metadata.open(newline="") as stream:
        for row in csv.DictReader(stream):
            group = group_for(row)
            if group is not None:
                desc = row["content_short_description"]
                candidates[group].setdefault(desc, []).append(row)

    selected = {}
    for group, count in GROUPS:
        by_desc = candidates[group]
        for rows in by_desc.values():
            rows.sort(key=lambda row: row["move_g1_path"])
        descriptions = sorted(by_desc)
        chosen = []
        # Round-robin descriptions to avoid filling a quota with one action/actor.
        while len(chosen) < count:
            added = False
            for desc in descriptions:
                if by_desc[desc]:
                    chosen.append(by_desc[desc].pop(0))
                    added = True
                    if len(chosen) == count:
                        break
            if not added:
                raise ValueError(f"Only {len(chosen)} eligible motions for {group}; need {count}")
        selected[group] = chosen
    return selected


def extract(archive: Path, dest: Path, selected: dict[str, list[dict[str, str]]]) -> None:
    wanted = {row["move_g1_path"] for rows in selected.values() for row in rows}
    found = set()
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            if member.name not in wanted:
                continue
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive member: {member.name}")
            target = dest.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            fileobj = source.extractfile(member)
            if fileobj is None:
                raise ValueError(f"Cannot read {member.name}")
            with fileobj, target.open("wb") as output:
                shutil.copyfileobj(fileobj, output)
            found.add(member.name)
            if len(found) % 50 == 0:
                print(f"Extracted {len(found)}/{len(wanted)}", flush=True)
    if found != wanted:
        raise ValueError(f"Missing {len(wanted - found)} selected CSVs from {archive}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/bones_seed/metadata/seed_metadata_v004.csv")
    )
    parser.add_argument("--archive", type=Path, default=Path("data/bones_seed/g1.tar.gz"))
    parser.add_argument("--dest", type=Path, default=Path("data/bones_seed/s0_source"))
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    selected = select(args.metadata)
    args.dest.mkdir(parents=True, exist_ok=True)
    manifest = args.dest / "S0_selection.json"
    manifest.write_text(json.dumps(selected, indent=2) + "\n")
    print({group: len(rows) for group, rows in selected.items()}, flush=True)
    if not args.select_only:
        extract(args.archive, args.dest, selected)
    print(f"Selection manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
