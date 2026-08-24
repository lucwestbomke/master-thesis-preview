# Conditions

One YAML per cell of the experiment matrix. The trainer reads none of these yet
-- `src/training/train.py` takes flags -- so they are the *record* of what a
reported run was, and the thing `docs/THESIS_PLAN.md` §3's 45 runs are
enumerated from.

Two invariants, both of which protect RQ1 rather than the learning:

* **`curriculum` is byte-identical in every file.** The schedule must be the
  same in every fidelity condition; if two files here disagree, the primary
  result is confounded and no analysis can recover it.
* **`reward` appears nowhere.** The reward is byte-identical across rungs
  (`src/env/test_fidelity.py` asserts it). Only `lambda_var` is ever swept, and
  it gets its own files when that ablation runs.
