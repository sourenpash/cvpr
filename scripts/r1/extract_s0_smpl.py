#!/usr/bin/env python3
"""Extract only kept S0 SMPL motions from NVIDIA's split, uncompressed tar archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transfer-report", type=Path, default=Path("data/motion_lib_r1/S0/transfer_report.json")
    )
    parser.add_argument(
        "--parts-dir", type=Path, default=Path("data/sonic_smpl_source/bones_seed_smpl")
    )
    parser.add_argument("--output", type=Path, default=Path("data/smpl_filtered/S0"))
    args = parser.parse_args()

    report = json.loads(args.transfer_report.read_text())
    names = {motion["name"] for motion in report["motions"] if not motion["dropped"]}
    if not names or any(name in (".", "..") or Path(name).name != name for name in names):
        raise ValueError("Transfer report has no kept motions or an unsafe motion name")
    wanted = {f"smpl_filtered/{name}.pkl": name for name in names}
    parts = sorted(args.parts_dir.glob("bones_seed_smpl.tar.part_*"))
    if len(parts) != 7:
        raise ValueError(f"Expected 7 SMPL tar parts, found {len(parts)} in {args.parts_dir}")

    args.output.mkdir(parents=True, exist_ok=True)
    found = set()
    with subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE) as cat:
        if cat.stdout is None:
            raise RuntimeError("Cannot read split tar stream")
        with tarfile.open(fileobj=cat.stdout, mode="r|") as archive:
            for member in archive:
                name = wanted.get(member.name)
                if name is None:
                    continue
                if not member.isfile():
                    raise ValueError(f"Expected a file at {member.name}")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Cannot read {member.name}")
                with source, (args.output / f"{name}.pkl").open("wb") as target:
                    shutil.copyfileobj(source, target)
                found.add(name)
                if len(found) % 50 == 0:
                    print(f"Extracted {len(found)}/{len(names)} SMPL motions", flush=True)
        cat.stdout.close()
        if cat.wait() != 0:
            raise RuntimeError("Failed to concatenate SMPL archive parts")
    if found != names:
        raise ValueError(f"Missing {len(names - found)} SMPL matches for kept R1 motions")
    print(f"SMPL subset: {len(found)} aligned motions in {args.output}", flush=True)


if __name__ == "__main__":
    main()
