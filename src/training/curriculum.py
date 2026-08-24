"""The curriculum schedule -- what moves `stage_weights` during a run.

`docs/ENVIRONMENT.md` specifies the four stages and `core.STAGES` implements
them; this is the thing that decides which of them episodes are drawn from as
training proceeds. Two rules, and both protect the *results* rather than the
learning:

1. **A fixed schedule by step count, in every reported run.** Adaptive
   advancement lets the easier fidelity rungs progress faster and hands them
   more experience at the final stage, which confounds RQ1 directly and
   unrecoverably. Use adaptive advancement in development to *find* a schedule;
   freeze it and run the same one everywhere.
2. **Mix in earlier stages (~20 % of episodes)** rather than hard-switching, or
   the policy forgets the opening it still has to execute every episode. That is
   why `stage_weights` is a weight vector and not a stage index.

The defence of rule 1 here is structural rather than procedural: `weights()` is
a pure function of **training progress alone**. It cannot read the environment,
the fidelity rung, the reward or the return, because it is never given them. A
test asserts the emitted sequence is identical across all five rungs at a fixed
seed, but the stronger guarantee is that there is no channel through which it
could differ.

⛔ Fidelity is never a curriculum axis. It is RQ1's independent variable; the
same reasoning forbids ramping building density.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..env.core import STAGES, BatchedSwarmEnv


@dataclass(frozen=True)
class CurriculumSchedule:
    """Fractions of the run at which the focus moves to the next stage.

    ⚠️ The default is **provisional** -- it is a starting point for the
    development sweep BLOCK_G.md §G4 licenses, not a measured schedule. Once the
    sweep has chosen one it is frozen here and every reported run uses it
    unchanged. Record the freeze in `docs/DECISIONS.md` when it happens.

    `boundaries[i]` is the progress at which stage `i+2` becomes the focus, so
    the default runs stage 1 for the first 15 % of training, stage 2 to 35 %,
    stage 3 to 60 %, and stage 4 for the remaining 40 %.
    """

    boundaries: tuple[float, ...] = (0.15, 0.35, 0.60)
    #: Share of episodes drawn from *earlier* stages once past stage 1.
    mix: float = 0.2

    def __post_init__(self) -> None:
        if len(self.boundaries) != len(STAGES) - 1:
            raise ValueError(
                f"expected {len(STAGES) - 1} boundaries for {len(STAGES)} stages, "
                f"got {len(self.boundaries)}"
            )
        if list(self.boundaries) != sorted(self.boundaries):
            raise ValueError(f"boundaries must be non-decreasing, got {self.boundaries}")
        if not 0.0 <= self.mix < 1.0:
            raise ValueError(f"mix must be in [0, 1), got {self.mix}")

    def focus(self, progress: float) -> int:
        """Index of the stage being taught at `progress` in [0, 1]."""
        return sum(1 for b in self.boundaries if progress >= b)

    def weights(self, progress: float) -> tuple[float, ...]:
        """Sampling weights over `STAGES`. A pure function of progress."""
        k = self.focus(min(max(progress, 0.0), 1.0))
        w = [0.0] * len(STAGES)
        if k == 0:
            w[0] = 1.0
            return tuple(w)
        w[k] = 1.0 - self.mix
        for i in range(k):
            w[i] = self.mix / k
        return tuple(w)


class CurriculumCallback:
    """Applies a `CurriculumSchedule` to a live env, once per training step.

    Cheap by design: it only touches the env when the weights actually change,
    so the hot loop pays one float comparison per step and a tiny tensor write
    at three boundaries in a whole run. Reweighting takes effect at the next
    auto-reset per environment, so episodes in flight keep the stage they were
    drawn under -- the transition is a change in the mix, not a discontinuity
    inside an episode.
    """

    def __init__(
        self,
        env: BatchedSwarmEnv,
        total_timesteps: int,
        schedule: CurriculumSchedule | None = None,
    ):
        if total_timesteps <= 0:
            raise ValueError(f"total_timesteps must be positive, got {total_timesteps}")
        self.env = env
        self.total_timesteps = total_timesteps
        self.schedule = schedule or CurriculumSchedule()
        self.current: tuple[float, ...] | None = None

    def progress(self, timestep: int) -> float:
        return timestep / self.total_timesteps

    def update(self, timestep: int) -> tuple[float, ...]:
        """Set the env's stage weights for `timestep`. Returns them, for logging."""
        want = self.schedule.weights(self.progress(timestep))
        if want != self.current:
            self.env.set_stage_weights(want)
            self.current = want
        return want
