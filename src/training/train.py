"""The training entrypoint: fidelity x architecture x seed.

    uv run python -m src.training.train --fidelity F4 --arch mlp --seed 0 \
        --env-steps 2_000_000 --num-envs 256 --device mps

One run is one `(fidelity, architecture, seed)` cell of the matrix. Fidelity is
fixed at construction and never changes during a run; the *curriculum* is what
varies during a run, and it runs the identical fixed schedule in every fidelity
condition (`curriculum.py`). Confusing the two is the one mistake that would
confound RQ1 past repair.

Three things about the loop are not skrl's defaults and two of them fail
silently -- `time_limit_bootstrap`, `discount_factor` and `value_preprocessor`,
all applied by `skrl_wrapper.mappo_cfg()`. A fourth is not a config setting at
all: the transition is recorded against the **pre-reset** observation and state
(`env.final_observations()` / `final_states()`), because with auto-reset the
tensors `step()` returns are already a fresh episode's opening and bootstrapping
off them values an unrelated state.

Units. `--env-steps` counts **transitions** -- one environment advancing one tick,
summed across the batch -- which is the unit `docs/BLOCK_D.md` settled the
throughput gate in and the unit THESIS_PLAN's 10 M-step budget is written in. The
number of batched iterations is `env_steps / num_envs`, and it is reported so
the two readings can never be confused again.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from skrl.memories.torch import RandomMemory
from skrl.multi_agents.torch.mappo import MAPPO

from ..env.core import GAMMA, BatchedSwarmEnv, EnvConfig, Fidelity
from ..models import (
    ARCHITECTURES,
    SwarmActor,
    SwarmActorRNN,
    SwarmCritic,
    SwarmCriticRNN,
    parameter_count,
)
from .curriculum import CurriculumCallback, CurriculumSchedule
from .skrl_wrapper import SWARM_UID, SharedPolicyWrapper, mappo_cfg, ppo_cfg

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


@dataclass
class TrainConfig:
    """One cell of the matrix, plus the learner settings.

    Everything here is a *learning* knob. ⛔ Nothing in the reward is reachable
    from this file except through `--tau-clearance` / `--tau-capacity` /
    `--potential-scale`, which live in the potential and so cannot move the
    optimum (PBRS), and `--lambda-var`, which is the one weight the design
    permits sweeping. Every other reward weight is pinned by the behavioural
    orderings in `docs/REWARD.md` and asserted in `test_reward.py`.
    """

    fidelity: Fidelity = "F4"
    architecture: str = "mlp"
    seed: int = 0
    num_envs: int = 256
    num_drones: int = 5
    env_steps: int = 2_000_000
    device: str = "cpu"

    # learner
    rollouts: int = 32
    learning_epochs: int = 4
    mini_batches: int = 4
    learning_rate: float = 3e-4
    entropy_loss_scale: float = 0.0
    min_log_std: float = -20.0  # skrl's default: effectively no exploration floor
    # Recurrent actor. The measured reason is in `SwarmActorRNN`'s docstring:
    # a feedforward actor cannot represent "I am the observer", and observer
    # tenure is where every learned policy loses to B0 by ~8x.
    # ⚠️ Yu et al. (2022)'s agent-specific global state -- their single largest
    # MAPPO recommendation, and the configuration this project has NOT been
    # running. Measured deficit it addresses (`scripts/probe_credit.py`): with
    # one global state repeated per drone, `max |V_i - V_j| = 0.000e+00` and
    # 0.015-0.06 % of advantage variance distinguishes one drone from another, so
    # the gradient carries no per-drone credit at all.
    agent_specific_critic: bool = False
    recurrent: bool = False
    # ⚠️ A recurrent actor implies a recurrent CRITIC, which is the published
    # MAPPO configuration (Yu et al., 2022) and a correctness argument rather
    # than a knob -- see `SwarmCriticRNN`. `--ff-critic` recovers the
    # feedforward critic, and exists so the A/B stays measurable rather than
    # because it is a defensible setting.
    recurrent_critic: bool = True
    rnn_hidden: int = 128
    sequence_length: int = 16
    hidden: int | None = None
    # KL-adaptive learning rate. PPO's usual answer to "it climbs, then
    # collapses": the update that destroys the policy is the one that moves the
    # distribution too far, and this shrinks the step exactly when that happens.
    # 0.0 disables it and uses a fixed learning rate.
    kl_threshold: float = 0.0

    # curriculum. `stage_weights` pins the run to a fixed mix and disables the
    # schedule -- G2's toy run uses it to sit on stage 1 alone.
    stage_weights: tuple[float, ...] | None = None
    schedule: CurriculumSchedule = field(default_factory=CurriculumSchedule)

    # reward knobs that PBRS proves are safe (learning speed only)
    tau_clearance_m: float | None = None
    tau_capacity_mbps: float | None = None
    potential_scale: float | None = None
    # Also inside Phi, so also optimum-preserving by the PBRS proof -- it sets
    # where `Phi_approach` has gradient. At the shipped 1500 m (the map diagonal)
    # the term is nearly flat across the 80-300 m band the policy actually
    # operates in: 8 m of closing pays 0.013. Measured stall distance under the
    # first full-mission pilot was 291 m against B0's 79 m.
    d_ref_m: float | None = None
    # The "hold" factor on Phi_observe -- also inside Phi, so also
    # optimum-preserving. It is the one knob aimed at the measured deficit
    # (observer tenure 47 against B0's 265): every reward term is FLAT while the
    # swarm is succeeding, so nothing distinguishes holding the sightline from
    # drifting out of it. 0.0 = the shipped potential, bitwise.
    w_hold: float | None = None
    d_hold_m: float | None = None
    # The per-drone relay potential -- the ONLY per-drone term in the reward, and
    # the one intervention aimed directly at the measured credit-assignment
    # deficit (`scripts/probe_credit.py`). PBRS-safe, including in the
    # multi-agent case (Devlin & Kudenko 2011), so it cannot move the optimum.
    w_relay: float | None = None
    lambda_var: float | None = None

    log_every: int = 20
    # Periodic checkpoints, so a run that peaks mid-training is not thrown away.
    # ⚠️ Selecting among them is model selection: do it on the TRAIN route split,
    # never on the eval split (`eval_policy.py` defaults to eval for exactly that
    # reason, and B0 paid a measured 0.6 pp for the same restriction).
    checkpoint_every: int = 0
    run_name: str | None = None
    wandb: bool = False
    # ⚠️ Tests only, and it is NOT a fidelity rung: it removes buildings from the
    # WORLD, sensor and diagnostics included (docs/BLOCK_F.md decision 7).
    # Occlusion is ~37x the rest of the step on CPU, so the loop test would take
    # minutes without it. A run that reports a number must never set this.
    no_buildings: bool = False

    @property
    def iterations(self) -> int:
        return max(self.env_steps // self.num_envs, 1)


def build_weights(cfg: TrainConfig):
    """`RewardWeights` with only the safe knobs moved."""
    from dataclasses import replace

    from ..env.reward import DEFAULT_WEIGHTS

    changes: dict[str, float] = {}
    if cfg.tau_clearance_m is not None:
        changes["tau_clearance_m"] = cfg.tau_clearance_m
    if cfg.tau_capacity_mbps is not None:
        changes["tau_capacity_mbps"] = cfg.tau_capacity_mbps
    if cfg.potential_scale is not None:
        changes["potential_scale"] = cfg.potential_scale
    if cfg.d_ref_m is not None:
        changes["d_ref_m"] = cfg.d_ref_m
    if cfg.w_hold is not None:
        changes["w_hold"] = cfg.w_hold
    if cfg.d_hold_m is not None:
        changes["d_hold_m"] = cfg.d_hold_m
    if cfg.w_relay is not None:
        changes["w_relay"] = cfg.w_relay
    if cfg.lambda_var is not None:
        changes["battery_variance"] = cfg.lambda_var
    return replace(DEFAULT_WEIGHTS, **changes) if changes else DEFAULT_WEIGHTS


def _kl_scheduler(agents: list[str], kl_threshold: float) -> dict:
    if kl_threshold <= 0.0:
        return {}
    from skrl.resources.schedulers.torch import KLAdaptiveLR

    return {
        "learning_rate_scheduler": KLAdaptiveLR,
        "learning_rate_scheduler_kwargs": {uid: {"kl_threshold": kl_threshold} for uid in agents},
    }


def build(cfg: TrainConfig) -> tuple[SharedPolicyWrapper, MAPPO, CurriculumCallback]:
    core = BatchedSwarmEnv(
        EnvConfig(
            num_envs=cfg.num_envs,
            num_drones=cfg.num_drones,
            device=cfg.device,
            seed=cfg.seed,
            fidelity=cfg.fidelity,
            training_extras=True,  # `final_state` + the per-term reward log
            auto_reset=True,
            stage_weights=cfg.stage_weights or (1.0, 0.0, 0.0, 0.0),
            no_buildings=cfg.no_buildings,
            compile_occlusion=cfg.device != "cpu" and not cfg.no_buildings,
        ),
        weights=build_weights(cfg),
    )
    env = SharedPolicyWrapper(core, agent_specific_state=cfg.agent_specific_critic)
    dev = env.device
    torch.manual_seed(cfg.seed)

    actor_kwargs = {
        "architecture": cfg.architecture,
        "hidden": cfg.hidden,
        "min_log_std": cfg.min_log_std,
    }
    if cfg.recurrent:
        actor = SwarmActorRNN(
            env.observation_spaces[SWARM_UID],
            env.action_spaces[SWARM_UID],
            dev,
            num_envs=env.num_envs,
            rnn_hidden=cfg.rnn_hidden,
            sequence_length=cfg.sequence_length,
            **actor_kwargs,
        ).to(dev)
    else:
        actor = SwarmActor(
            env.observation_spaces[SWARM_UID],
            env.action_spaces[SWARM_UID],
            dev,
            **actor_kwargs,
        ).to(dev)
    if cfg.recurrent and cfg.recurrent_critic:
        critic = SwarmCriticRNN(
            env.state_spaces[SWARM_UID],
            env.action_spaces[SWARM_UID],
            dev,
            num_envs=env.num_envs,
            rnn_hidden=cfg.rnn_hidden,
            sequence_length=cfg.sequence_length,
        ).to(dev)
    else:
        critic = SwarmCritic(env.state_spaces[SWARM_UID], env.action_spaces[SWARM_UID], dev).to(dev)

    memory = RandomMemory(memory_size=cfg.rollouts, num_envs=env.num_envs, device=dev)
    if cfg.recurrent:
        # ⚠️ NOT skrl's PPO_RNN: it stores the post-step hidden state as the
        # pre-step one for every transition after the first. See recurrent_ppo.
        from .recurrent_ppo import PPO_RNN_Aligned

        agent = PPO_RNN_Aligned(
            models={"policy": actor, "value": critic},
            memory=memory,
            observation_space=env.observation_spaces[SWARM_UID],
            state_space=env.state_spaces[SWARM_UID],
            action_space=env.action_spaces[SWARM_UID],
            cfg=ppo_cfg(
                device=dev,
                rollouts=cfg.rollouts,
                learning_epochs=cfg.learning_epochs,
                mini_batches=cfg.mini_batches,
                learning_rate=cfg.learning_rate,
                entropy_loss_scale=cfg.entropy_loss_scale,
            ),
            device=dev,
        )
        curriculum = (
            _FixedStages(core, cfg.stage_weights)
            if cfg.stage_weights is not None
            else CurriculumCallback(core, cfg.iterations, cfg.schedule)
        )
        return env, agent, curriculum

    models = {SWARM_UID: {"policy": actor, "value": critic}}
    memories = {SWARM_UID: memory}
    agent = MAPPO(
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,
        cfg=mappo_cfg(
            env.possible_agents,
            device=dev,
            rollouts=cfg.rollouts,
            learning_epochs=cfg.learning_epochs,
            mini_batches=cfg.mini_batches,
            learning_rate=cfg.learning_rate,
            entropy_loss_scale=cfg.entropy_loss_scale,
            **_kl_scheduler(env.possible_agents, cfg.kl_threshold),
        ),
        observation_spaces=env.observation_spaces,
        state_spaces=env.state_spaces,
        action_spaces=env.action_spaces,
        device=dev,
    )
    curriculum = CurriculumCallback(core, cfg.iterations, cfg.schedule)
    if cfg.stage_weights is not None:
        curriculum = _FixedStages(core, cfg.stage_weights)
    return env, agent, curriculum


class _FixedStages(CurriculumCallback):
    """`--stage-weights` : no schedule at all, one fixed mix for the whole run.

    Development and pilot use only (G2 sits on stage 1 with this). A reported
    run must walk the schedule, identically in every fidelity condition.
    """

    def __init__(self, env: BatchedSwarmEnv, weights: tuple[float, ...]):
        super().__init__(env, total_timesteps=1)
        self.fixed = tuple(weights)

    def update(self, timestep: int) -> tuple[float, ...]:
        if self.current != self.fixed:
            self.env.set_stage_weights(self.fixed)
            self.current = self.fixed
        return self.fixed


class Meter:
    """On-device accumulators. One host sync per logging interval, never per step.

    `AGENTS.md` forbids `.item()` in the hot loop, and it is the easy rule to
    break while adding instrumentation -- which is why every quantity here is a
    running tensor and the only `.tolist()` is in `drain`.
    """

    KEYS = ("mission_capable", "observed", "capacity", "hop_count", "chain_occluded", "reward")

    def __init__(self, device: torch.device | str):
        self.device = device
        self.sums = torch.zeros(len(self.KEYS), device=device)
        self.terms: dict[str, torch.Tensor] = {}
        self.count = 0
        self.episode_return = None
        self.finished_return = torch.zeros((), device=device)
        self.finished = torch.zeros((), device=device)

    def add(self, rew: torch.Tensor, extras: dict[str, torch.Tensor], done: torch.Tensor) -> None:
        """`rew` is `(B, N)` -- the env's own shape, not skrl's `(B*N, 1)` view."""
        values = torch.stack(
            [
                extras["mission_capable"].float().mean(),
                extras["sees_any"].float().mean(),
                extras["e2e_capacity_mbps"].mean(),
                extras["hop_count"].float().mean(),
                extras["chain_occluded"].float().mean(),
                rew.mean(),
            ]
        )
        self.sums += values
        for key, value in extras.items():
            if key.startswith("reward/"):
                self.terms[key] = (
                    self.terms.get(key, torch.zeros((), device=self.device)) + value.mean()
                )
        if self.episode_return is None:
            self.episode_return = torch.zeros_like(rew.mean(dim=-1))
        self.episode_return = self.episode_return + rew.mean(dim=-1)
        self.finished_return += (self.episode_return * done).sum()
        self.finished += done.sum()
        self.episode_return = self.episode_return * (~done).float()
        self.count += 1

    def drain(self) -> dict[str, float]:
        """Reduce and reset. The single host sync."""
        if self.count == 0:
            return {}
        names = list(self.KEYS) + list(self.terms) + ["episode_return", "episodes"]
        packed = torch.cat(
            [
                self.sums / self.count,
                torch.stack([v / self.count for v in self.terms.values()])
                if self.terms
                else torch.zeros(0, device=self.device),
                (self.finished_return / self.finished.clamp_min(1)).reshape(1),
                self.finished.reshape(1),
            ]
        )
        out = dict(zip(names, packed.tolist(), strict=True))
        self.sums.zero_()
        for key in self.terms:
            self.terms[key] = torch.zeros((), device=self.device)
        self.finished_return.zero_()
        self.finished.zero_()
        self.count = 0
        return out


def train(cfg: TrainConfig) -> Path:
    env, agent, curriculum = build(cfg)
    agent.init()
    agent.enable_training_mode(True)

    run = cfg.run_name or f"{cfg.fidelity}-{cfg.architecture}-s{cfg.seed}"
    out = RUNS_DIR / run
    out.mkdir(parents=True, exist_ok=True)
    logger = _open_log(cfg, out, env, agent)

    obs, _ = env.reset()
    states = env.state()
    meter = Meter(env.device)
    started = time.perf_counter()

    # PPO_RNN is single-agent and takes plain tensors; MAPPO takes agent-keyed
    # dicts. `SharedPolicyWrapper` produces one key either way, so the only
    # difference is whether it is unwrapped.
    one = SWARM_UID
    unwrap = (lambda d: d[one]) if cfg.recurrent else (lambda d: d)
    rewrap = (lambda t: {one: t}) if cfg.recurrent else (lambda t: t)

    for timestep in range(cfg.iterations):
        weights = curriculum.update(timestep)
        agent.pre_interaction(timestep=timestep, timesteps=cfg.iterations)
        with torch.no_grad():
            actions = agent.act(
                unwrap(obs), unwrap(states), timestep=timestep, timesteps=cfg.iterations
            )[0]
        next_obs, rewards, terminated, truncated, infos = env.step(rewrap(actions))

        # BEFORE `record_transition`, which adds the truncation bootstrap into
        # `rewards` in place. Logging afterwards would report a return the
        # environment never paid.
        extras = infos[SWARM_UID]
        done = (terminated[SWARM_UID] | truncated[SWARM_UID]).view(env.core.cfg.num_envs, -1)[:, 0]
        meter.add(rewards[SWARM_UID].view(env.core.cfg.num_envs, -1), extras, done)

        # ⚠️ NOT `next_obs` / `env.state()`. With auto_reset those are already a
        # fresh episode's opening wherever an episode ended, and skrl's
        # truncation bootstrap would value an unrelated state. See
        # `SharedPolicyWrapper.final_observations`.
        agent.record_transition(
            observations=unwrap(obs),
            states=unwrap(states),
            actions=actions,
            rewards=unwrap(rewards),
            next_observations=unwrap(env.final_observations()),
            next_states=unwrap(env.final_states()),
            terminated=unwrap(terminated),
            truncated=unwrap(truncated),
            infos=infos,
            timestep=timestep,
            timesteps=cfg.iterations,
        )
        agent.post_interaction(timestep=timestep, timesteps=cfg.iterations)

        if cfg.checkpoint_every and (timestep + 1) % cfg.checkpoint_every == 0:
            _save(cfg, agent, out / f"checkpoint-{(timestep + 1) * cfg.num_envs}.pt")

        if (timestep + 1) % cfg.log_every == 0:
            row = meter.drain()
            row.update(_drain_tracking(agent))
            row["timestep"] = timestep + 1
            row["env_steps"] = (timestep + 1) * cfg.num_envs
            row["stage_focus"] = float(max(range(len(weights)), key=lambda i: weights[i]))
            row["env_steps_per_s"] = row["env_steps"] / (time.perf_counter() - started)
            logger(row)

        obs, states = next_obs, env.state()

    path = _save(cfg, agent, out / "checkpoint.pt")
    elapsed = time.perf_counter() - started
    steps = cfg.iterations * cfg.num_envs
    print(
        f"\n{run}: {steps:,} env-steps in {elapsed / 60:.1f} min "
        f"({steps / elapsed:,.0f} env-steps/s) -> {path}"
    )
    print(f"  10M-step projection: {10e6 / (steps / elapsed) / 3600:.2f} h (target <=3 h)")
    return path


def _policy_of(agent):
    """MAPPO keys models by agent id; PPO_RNN holds them directly."""
    return agent.policies[SWARM_UID] if hasattr(agent, "policies") else agent.policy


def _value_of(agent):
    return agent.values[SWARM_UID] if hasattr(agent, "values") else agent.value


def _save(cfg: TrainConfig, agent: MAPPO, path: Path) -> Path:
    torch.save(
        {
            "config": {k: v for k, v in asdict(cfg).items() if k != "schedule"},
            "schedule": asdict(cfg.schedule),
            "architecture": cfg.architecture,
            "hidden": cfg.hidden,
            "recurrent": cfg.recurrent,
            "rnn_hidden": cfg.rnn_hidden,
            "sequence_length": cfg.sequence_length,
            "policy": _policy_of(agent).state_dict(),
            "value": _value_of(agent).state_dict(),
            "env_steps": cfg.iterations * cfg.num_envs,
            "gamma": GAMMA,
        },
        path,
    )
    return path


def _drain_tracking(agent: MAPPO) -> dict[str, float]:
    """skrl's own diagnostics: policy loss, value loss, action std.

    Free -- the agent accumulates them during `update` regardless -- and they are
    the first place to look when the return curve is flat or falling. A rising
    value loss with a collapsing standard deviation is a different disease from
    a flat policy loss.
    """
    rows = {}
    for tag, values in agent.tracking_data.items():
        if values:
            key = tag.split(" (")[0].replace("Loss / ", "loss/").replace("Policy / ", "policy/")
            rows[key.replace(" ", "_").lower()] = sum(values) / len(values)
    agent.tracking_data.clear()
    return rows


def _open_log(cfg: TrainConfig, out: Path, env, agent):
    """JSONL always; W&B only when asked, so a run never needs the network."""
    handle = (out / "log.jsonl").open("w")
    meta = {
        "config": {k: v for k, v in asdict(cfg).items() if k != "schedule"},
        "schedule": asdict(cfg.schedule),
        "iterations": cfg.iterations,
        "actor_parameters": parameter_count(_policy_of(agent)),
        "critic_parameters": parameter_count(_value_of(agent)),
        "rows_per_update": cfg.rollouts * env.num_envs,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

    wb = None
    if cfg.wandb:
        import wandb as wb_module

        wb = wb_module
        wb.init(project="uav-swarm-marl", name=out.name, config=meta)

    def log(row: dict[str, Any]) -> None:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        if wb is not None:
            wb.log(row, step=int(row["timestep"]))
        print(
            f"  it {row['timestep']:>6}  steps {row['env_steps']:>10,}  "
            f"capable {row['mission_capable'] * 100:5.1f} %  "
            f"observed {row['observed'] * 100:5.1f} %  "
            f"ret/ep {row['episode_return']:8.1f}  "
            f"{row['env_steps_per_s']:,.0f} steps/s"
        )

    return log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fidelity", default="F4", choices=["F0", "F1", "F2", "F3", "F4"])
    ap.add_argument("--arch", default="mlp", choices=list(ARCHITECTURES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--num-drones", type=int, default=5)
    ap.add_argument("--env-steps", type=int, default=2_000_000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--rollouts", type=int, default=32)
    ap.add_argument("--learning-epochs", type=int, default=4)
    ap.add_argument("--mini-batches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy", type=float, default=0.0)
    ap.add_argument("--min-std", type=float, default=None, help="floor on the action std")
    ap.add_argument(
        "--agent-specific-critic",
        action="store_true",
        help="append each drone's own ego block to the critic state (Yu et al. 2022). "
        "Without it V_i is bit-identical across drones and the gradient carries no "
        "per-drone credit -- see scripts/probe_credit.py",
    )
    ap.add_argument("--recurrent", action="store_true", help="GRU actor (skrl PPO_RNN)")
    ap.add_argument(
        "--ff-critic",
        action="store_true",
        help="feedforward critic under a recurrent actor (the A/B control, not a setting)",
    )
    ap.add_argument("--rnn-hidden", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--kl", type=float, default=0.0, help="KL-adaptive LR threshold (0 = off)")
    # Development knobs for FINDING the schedule. BLOCK_G.md licenses searching
    # for it and then freezing it; a reported run uses the frozen default in
    # every fidelity condition, or RQ1 is confounded.
    ap.add_argument("--boundaries", type=float, nargs=3, default=None)
    ap.add_argument("--mix", type=float, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument(
        "--stage",
        type=int,
        default=None,
        help="pin the whole run to one curriculum stage (1-4). Pilots only.",
    )
    ap.add_argument("--tau-clearance", type=float, default=None)
    ap.add_argument("--tau-capacity", type=float, default=None)
    ap.add_argument("--potential-scale", type=float, default=None)
    ap.add_argument("--d-ref", type=float, default=None)
    ap.add_argument(
        "--w-hold", type=float, default=None, help="Phi_observe hold factor; 0 = shipped"
    )
    ap.add_argument(
        "--d-hold", type=float, default=None, help="range scale for the hold factor (m)"
    )
    ap.add_argument("--lambda-var", type=float, default=None)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--checkpoint-every", type=int, default=0, help="iterations; 0 = final only")
    ap.add_argument("--name", default=None)
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    stage_weights = None
    if a.stage is not None:
        stage_weights = tuple(1.0 if i == a.stage - 1 else 0.0 for i in range(4))
    schedule = CurriculumSchedule(
        **{
            k: v
            for k, v in (
                ("boundaries", tuple(a.boundaries) if a.boundaries else None),
                ("mix", a.mix),
            )
            if v is not None
        }
    )

    train(
        TrainConfig(
            fidelity=a.fidelity,
            architecture=a.arch,
            seed=a.seed,
            num_envs=a.num_envs,
            num_drones=a.num_drones,
            env_steps=a.env_steps,
            device=device,
            rollouts=a.rollouts,
            learning_epochs=a.learning_epochs,
            mini_batches=a.mini_batches,
            learning_rate=a.lr,
            entropy_loss_scale=a.entropy,
            min_log_std=math.log(a.min_std) if a.min_std else -20.0,
            agent_specific_critic=a.agent_specific_critic,
            recurrent=a.recurrent,
            recurrent_critic=not a.ff_critic,
            rnn_hidden=a.rnn_hidden,
            sequence_length=a.seq_len,
            kl_threshold=a.kl,
            hidden=a.hidden,
            stage_weights=stage_weights,
            schedule=schedule,
            tau_clearance_m=a.tau_clearance,
            tau_capacity_mbps=a.tau_capacity,
            potential_scale=a.potential_scale,
            d_ref_m=a.d_ref,
            w_hold=a.w_hold,
            d_hold_m=a.d_hold,
            w_relay=a.w_relay,
            lambda_var=a.lambda_var,
            log_every=a.log_every,
            checkpoint_every=a.checkpoint_every,
            run_name=a.name,
            wandb=a.wandb,
        )
    )


if __name__ == "__main__":
    main()
