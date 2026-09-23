#!/usr/bin/env python3
"""Fetch the upstream Unitree R1 / Dex3-1 robot-description sources (pinned commits).

Downloads only the small robot-description files (URDF/MJCF/STL) that
``build_r1_assets.py`` transforms into SONIC assets. It does NOT download any
motion data or checkpoints.

Everything lands in ``third_party_assets/`` (git-ignored). Re-running is
idempotent (existing files are skipped unless ``--force``).

Usage:
    python scripts/r1/fetch_upstream_assets.py            # fetch all
    python scripts/r1/fetch_upstream_assets.py --force    # re-download
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[2]
DEST_ROOT = REPO_ROOT / "third_party_assets"

# Pinned upstream commits (2026-09-23). Bump deliberately and re-run the asset tests.
SOURCES = {
    "unitree_ros": {
        "repo": "unitreerobotics/unitree_ros",
        "commit": "ccfc6fd8430a17ba3dacef9a1e2faf64ff3b0aee",
        "prefixes": [
            "robots/r1_description/",
            "robots/dexterous_hand_description/dex3_1/",
        ],
        "license": "BSD-3-Clause",
    },
    "unitree_rl_mjlab": {
        "repo": "unitreerobotics/unitree_rl_mjlab",
        "commit": "1425b15f73bd4095f0df53709d7c389c3eb9e790",
        "prefixes": ["src/assets/robots/unitree_r1/"],
        "license": "BSD-3-Clause",
    },
    "unitree_mujoco": {
        "repo": "unitreerobotics/unitree_mujoco",
        "commit": "1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d",
        "prefixes": ["unitree_robots/r1/"],
        "license": "BSD-3-Clause",
    },
}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "sonic-r1-fetch"})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
        return r.read()


def list_tree(repo: str, commit: str) -> list[str]:
    data = json.loads(_get(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1"))
    if data.get("truncated"):
        sys.exit(f"GitHub tree listing truncated for {repo}; add explicit file lists.")
    return [t["path"] for t in data["tree"] if t["type"] == "blob"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    ap.add_argument("--only", choices=sorted(SOURCES), nargs="*", help="subset of sources")
    args = ap.parse_args()

    manifest = {}
    for name, src in SOURCES.items():
        if args.only and name not in args.only:
            continue
        repo, commit = src["repo"], src["commit"]
        paths = [
            p for p in list_tree(repo, commit) if any(p.startswith(pre) for pre in src["prefixes"])
        ]
        dest_dir = DEST_ROOT / name
        print(f"[{name}] {repo}@{commit[:10]}: {len(paths)} files -> {dest_dir}")
        n_new = 0
        for p in paths:
            out = dest_dir / p
            if out.exists() and not args.force:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(_get(f"https://raw.githubusercontent.com/{repo}/{commit}/{p}"))
            n_new += 1
        print(f"[{name}] downloaded {n_new}, skipped {len(paths) - n_new}")
        manifest[name] = {"repo": repo, "commit": commit, "license": src["license"], "files": paths}

    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    (DEST_ROOT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {DEST_ROOT / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
