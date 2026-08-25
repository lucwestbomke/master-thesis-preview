"""Does the learner's gradient distinguish one drone from another?

    uv run python scripts/probe_credit.py --device cuda
    uv run python scripts/probe_credit.py --device cuda --checkpoint runs/g8-ff-shipped-s0/checkpoint.pt

## The question, and why it comes before any reward change

`docs/BLOCK_G.md` measures a role-emergence failure: an observer partly emerges,
a relay never does, and conditioned on observing the learned chain is
indistinguishable from a random policy's. The proposed fix is a per-drone
potential -- but that only makes sense if the learner *has a channel* for
per-drone credit in the first place.

It may not. Two structural facts about this setup:

* `SharedPolicyWrapper` hands skrl **one global state repeated per drone**
  (`repeat_interleave(n, dim=0)`), so `V(s)` is bit-identical across the `N`
  drones of an environment. Not approximately -- identically.
* The reward is team-dominated: `mission` 1.0, `idle` 0.3, `battery_variance`
  0.5 are team quantities; only `energy` 0.15 and `effort` 0.01 are per-drone,
  and both are symmetric costs.

If both hold, then `A_i = r_i + gamma*V(s') - V(s)` differs across drones only by
the small individual costs, and the policy gradient
`grad log pi(a_i | o_i) * A_i` tells every drone the same thing: *the team did
well*. Never *your action was the good one*. That is the multi-agent credit
assignment problem, unmitigated -- the one COMA (Foerster et al., 2018) exists
to solve.

## What it reports

A variance decomposition of the advantage over `(t, env, drone)`:

    between-drone share = E_{t,env}[ Var_i(A) ] / Var(A)

**0 means the gradient cannot tell one drone from another.** The same
decomposition is reported for rewards and values, so the cause is attributable
rather than inferred, plus the share of `|A|` that the per-drone reward terms
account for.

⚠️ This measures the *learner*, not the policy, so it is meaningful at
initialisation too -- the claim is architectural. `--checkpoint` loads trained
actor weights to confirm it under a realistic state distribution.

## 📏 Measured 2026-08-25 -- the channel is not there

| condition | advantage | reward | value |
|---|---|---|---|
| stage 4, GNN, untrained, buildings | **0.00041** | 0.00033 | **0.00000** |
| stage 1, MLP, **trained**, buildings | **0.00015** | 0.00024 | **0.00000** |
| stage 1, MLP, untrained, no buildings | **0.00059** | 0.50462 | **0.00000** |

**0.015-0.06 % of advantage variance distinguishes one drone from another**, and
`max |V_i - V_j|` is exactly `0.000e+00` in every condition -- as it must be,
since the critic is handed one state repeated `N` times.

So the policy gradient carries **no per-drone credit at all**. Whatever the swarm
does, all five drones are told the same thing. Role differentiation cannot be
learned from a signal that is constant across the agents it would differentiate,
which is why no amount of tuning, memory or team-quantity shaping moved it.

⚠️ Note the third row: without buildings, 50 % of the *reward* variance is
per-drone (the energy and effort costs), and the advantage still washes it out to
0.06 %. GAE accumulates the team component coherently across ~19 effective steps
while the per-drone costs largely cancel -- so even a per-drone reward term has to
clear that filter to reach the gradient.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skrl.multi_agents.torch.mappo.mappo import compute_gae

from src.training.skrl_wrapper import SWARM_UID
from src.training.train import TrainConfig, build


def _share(x: torch.Tensor) -> float:
    """`E_{t,env}[Var_i(x)] / Var(x)` for an `(T, B, N)` tensor.

    The fraction of total variance that lives *between drones at the same
    instant* -- i.e. the only part of the signal that can differentiate them.
    """
    total = x.var(unbiased=False)
    if float(total) == 0.0:
        return 0.0
    within = x.var(dim=2, unbiased=False).mean()
    return float(within / total)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="gnn")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--num-drones", type=int, default=5)
    ap.add_argument("--rollouts", type=int, default=32)
    ap.add_argument("--stage", type=int, default=4)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument(
        "--agent-specific-critic",
        action="store_true",
        help="Yu et al. (2022) agent-specific global state -- the intervention this probe motivates",
    )
    ap.add_argument("--w-relay", type=float, default=None, help="per-drone relay potential")
    ap.add_argument("--no-buildings", action="store_true", help="tests only; not a rung")
    a = ap.parse_args()

    cfg = TrainConfig(
        architecture=a.arch,
        num_envs=a.num_envs,
        num_drones=a.num_drones,
        device=a.device,
        rollouts=a.rollouts,
        env_steps=a.rollouts * a.num_envs,
        stage_weights=tuple(1.0 if i == a.stage - 1 else 0.0 for i in range(4)),
        no_buildings=a.no_buildings,
        agent_specific_critic=a.agent_specific_critic,
        w_relay=a.w_relay,
    )
    env, agent, curriculum = build(cfg)
    agent.init()
    agent.enable_training_mode(True)

    if a.checkpoint:
        blob = torch.load(a.checkpoint, map_location=env.device, weights_only=False)
        agent.models[SWARM_UID]["policy"].load_state_dict(blob["policy"])
        print(f"loaded actor from {a.checkpoint}")

    obs, _ = env.reset()
    states = env.state()
    for t in range(cfg.rollouts):
        curriculum.update(t)
        with torch.no_grad():
            actions = agent.act(obs, states, timestep=t, timesteps=cfg.rollouts)[0]
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        agent.record_transition(
            observations=obs,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=env.final_observations(),
            next_states=env.final_states(),
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=t,
            timesteps=cfg.rollouts,
        )
        obs, states = next_obs, env.state()

    memory = agent.memories[SWARM_UID]
    get = memory.get_tensor_by_name
    values = get("values")
    # GAE exactly as MAPPO computes it, including the truncation handling.
    with torch.no_grad():
        last, _ = agent.values[SWARM_UID].act(
            {
                "observations": agent._observation_preprocessor[SWARM_UID](
                    env.final_observations()[SWARM_UID]
                ),
                "states": agent._state_preprocessor[SWARM_UID](env.final_states()[SWARM_UID]),
            },
            role="value",
        )
        last = agent._value_preprocessor[SWARM_UID](last, inverse=True)
    _returns, advantages = compute_gae(
        rewards=get("rewards"),
        terminated=get("terminated"),
        truncated=get("truncated"),
        values=values,
        last_values=last,
        discount_factor=agent.cfg.discount_factor[SWARM_UID],
        lambda_coefficient=agent.cfg.gae_lambda[SWARM_UID],
        time_limit_bootstrap=agent.cfg.time_limit_bootstrap[SWARM_UID],
    )

    # (T, B*N, 1) -> (T, B, N). Rows are `b*N + i` because SharedPolicyWrapper
    # uses `repeat_interleave(n, dim=0)`; `test_skrl_wrapper.py` pins that.
    t, b, n = cfg.rollouts, a.num_envs, a.num_drones
    adv = advantages.view(t, b, n)
    rew = get("rewards").view(t, b, n)
    val = values.view(t, b, n)

    print(
        f"\nrollout {t} x {b} envs x {n} drones, arch={a.arch}, stage={a.stage}, "
        f"agent_specific_critic={a.agent_specific_critic}, w_relay={a.w_relay}\n"
    )
    print("variance that lives BETWEEN DRONES at the same instant")
    print("  (0 = the signal cannot tell one drone from another)\n")
    print(f"  advantage : {_share(adv):8.5f}")
    print(f"  reward    : {_share(rew):8.5f}")
    print(f"  value     : {_share(val):8.5f}")

    spread = (val.max(dim=2).values - val.min(dim=2).values).abs().max()
    print(f"\n  max |V_i - V_j| within an env, over the whole rollout: {float(spread):.3e}")
    print("  (exactly 0 is expected: the critic is handed ONE state, repeated)")

    a_rng = (adv.max(dim=2).values - adv.min(dim=2).values).abs()
    print(f"\n  mean across-drone advantage range : {float(a_rng.mean()):.5f}")
    print(f"  std of advantage across time      : {float(adv.std(unbiased=False)):.5f}")
    print(
        f"  ratio                             : {float(a_rng.mean() / adv.std(unbiased=False)):.5f}"
    )


if __name__ == "__main__":
    main()
