# Agent instructions for this repository

This is a fork of NVIDIA's `GR00T-WholeBodyControl` (GEAR-SONIC). Our project adds a
**Unitree R1 + Dex3-1** embodiment and fine-tunes SONIC to track end-effector pose targets.

**Start here:** read `PLAN.md`. Its Status ledger (§2) says what is done and what is next;
its task cards (§5) contain the exact commands and acceptance criteria. Update the ledger
when you finish or unblock a task.

## Ground rules

- Machine roles matter. Asset/data-tooling/tests run anywhere (conda env from
  `environment.yml`); anything needing Isaac Lab or datasets runs on the Linux GPU box.
  Never download datasets or checkpoints onto a machine that is not the training box.
- All R1 constants live in `gear_sonic/utils/embodiment/r1_spec.py`. Change them there,
  re-run `python scripts/r1/build_r1_assets.py`, then `pytest gear_sonic/tests/r1 -q`.
  Never hand-edit `r1_dex3.urdf`, `r1_dex3.xml` or `r1_ordering.py` (generated).
- Keep diffs to upstream files minimal and isolated (we merge `upstream/main` regularly).
  New tooling goes in `scripts/r1/`, tests in `gear_sonic/tests/r1/`, docs in `docs/r1/`.
- Robot-specific code must be selected by `robot.type` (`r1_dex3`), never by hard-coding
  G1 indices. If you find a new G1 hard-code, make it configurable with the G1 value as
  default and add it to PLAN.md §3.3.
- Values marked VERIFY in `r1_spec.py` / PLAN.md are estimates awaiting hardware/CAD
  confirmation; do not silently "fix" them.
- Tests: `python -m pytest gear_sonic/tests/r1 -q`. Lint touched files with
  `python -m ruff check <files>` and `python -m black <files>` (do not reformat upstream files wholesale).
- Before training on the GPU box, task **G2** (Isaac Lab ordering verification) must pass;
  it is the single check that protects against scrambled observations/actions.
