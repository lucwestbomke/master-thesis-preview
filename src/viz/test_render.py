"""The presentation renderer's argument handling.

Not a test of what the figures look like -- that is what looking at them is for.
This pins the one thing about `render_episode.py` that can fail *silently*: which
policies it actually flies. A renderer that draws the wrong policy still produces
five confident figures and a table of numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from render_episode import POLICIES, compare_set


def test_compare_keeps_a_checkpoint_instead_of_discarding_it():
    """`--compare --policy <ckpt>` must draw the checkpoint, not just baselines.

    ⚠️ It used to be `policies = POLICIES if a.compare else (a.policy,)`, which
    discarded `--policy` outright. So

        render_episode.py --policy runs/<name>/checkpoint.pt --compare --route 12

    -- the exact command `BLOCK_G.md` recommends for turning an aggregate into a
    mechanism -- rendered the five scripted baselines, reported success for all
    five, and never flew the checkpoint under investigation.
    """
    ckpt = "runs/g8-ff-shipped-s0/checkpoint.pt"
    assert compare_set(ckpt) == (*POLICIES, ckpt)
    assert ckpt in compare_set(ckpt), "the checkpoint is what the command exists to draw"


def test_compare_does_not_duplicate_a_named_baseline():
    for name in POLICIES:
        assert compare_set(name) == POLICIES


def test_the_baseline_set_is_ordered_worst_to_best():
    """The figures are read side by side, so the order carries meaning.

    `random` -> `waypoint` -> `b0-geodesic` -> `b0` sets up the comparison a
    reader makes anyway, and a learned checkpoint appends after them.
    """
    assert POLICIES[0] == "random"
    assert POLICIES.index("b0-geodesic") < POLICIES.index("b0")
