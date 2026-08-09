"""
Swarm relay-tracking environment, PettingZoo ParallelEnv API.

SCOPE: this is the *adapter*, not the training path. It exists for PettingZoo
API-compliance tests and single-env visual debugging only.

Training runs against a batched tensor core (leading `num_envs` dimension)
through a custom skrl multi-agent wrapper. Reason, verified against the
installed stack: skrl's `PettingZooWrapper` round-trips every action and
observation through NumPy on each step (`untensorize_space` /
`tensorize_space`) and exposes `num_envs == 1`, which contradicts the project's
stay-in-VRAM rule and caps throughput at single-env Python speed. See
AGENTS.md "Device / performance rules" and docs/THESIS_PLAN.md section 6.

Purely numeric/tensor state — no rendering happens here. See render.py for
the separate matplotlib top-down visualizer used for eval videos / figures.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import torch
from gymnasium import spaces
from pettingzoo import ParallelEnv

OBS_DIM = 13  # kinematics, battery, tracking metrics, ambient noise floor
# Motion only. Transmit power is fixed at 30 dBm -- adaptive Ptx was tested
# against fair baselines under three separate justifications (energy,
# interference, detectability) and came out null each time. See
# docs/NEGATIVE_RESULTS.md before adding a 4th dimension back.
ACTION_DIM = 3  # dv_x, dv_y, dv_z
PTX_FIXED_DBM = 30.0
CAPACITY_THRESHOLD_MBPS = 5.0
MAX_JAM_STEPS = 5  # consecutive steps link may be down before termination


class SwarmRelayEnv(ParallelEnv):
    metadata: ClassVar[dict] = {"name": "swarm_relay_v0", "render_modes": ["rgb_array"]}

    def __init__(self, num_drones: int = 5, city_data=None, device: str = "cpu"):
        """
        city_data: preprocessed OSM assets (building polygons/heights, road
        graph for HVT routing) — loaded once, shared read-only across
        episodes. Load via scripts/prep_osm.py, not inside __init__.
        """
        self.possible_agents = [f"tactical_node_{i}" for i in range(num_drones)]
        self.city_data = city_data
        self.device = device

        # Internal state lives as tensors, not per-agent Python objects —
        # keeps the door open for later GPU-batched vectorization without
        # a rewrite. Shapes below assume a single env instance (no leading
        # num_envs dim yet).
        self._positions: torch.Tensor | None = None  # (num_drones, 3)
        self._velocities: torch.Tensor | None = None  # (num_drones, 3)
        self._battery: torch.Tensor | None = None  # (num_drones,)
        self._ptx_dbm: torch.Tensor | None = None  # (num_drones,)
        self._hvt_position: torch.Tensor | None = None  # (3,)
        self._hvt_route: torch.Tensor | None = None  # (T, 3) waypoints
        self._step_count = 0
        self._link_down_streak = 0

        self.agents = []

        # Built once as plain dicts. PettingZoo 1.26 and skrl both read the
        # `observation_spaces` / `action_spaces` *attributes*; providing only
        # the accessor methods raises AttributeError on the first wrapped
        # step(). `functools.lru_cache` on the methods would also pin `self`
        # for the process lifetime.
        self.observation_spaces = {
            a: spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
            for a in self.possible_agents
        }
        # dv_x, dv_y, dv_z, scaled to acceleration limits downstream.
        self.action_spaces = {
            a: spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
            for a in self.possible_agents
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._step_count = 0
        self._link_down_streak = 0

        # TODO: sample a random start/end pair on self.city_data road graph,
        # build self._hvt_route (randomized per episode, not fixed)
        # TODO: initialize drone positions/velocities/battery=1.0/ptx_dbm

        observations = {a: self._build_observation(i) for i, a in enumerate(self.agents)}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions: dict[str, np.ndarray]):
        # 1. Apply actions -> update velocity/position (kinematics). Ptx is
        #    constant at PTX_FIXED_DBM; it is not an action.
        # 2. Advance HVT along its route by one step
        # 3. Update battery: P_total = P_flight(||v||) + kappa*||a||^2
        #    + P_tx_DC, where P_flight is the rotary-wing model (Zeng et al.
        #    2019, U-shaped in speed) and P_tx_DC is constant. battery -= P*dt
        # 4. Occlusion: batched torch segment-vs-box (slab method) against the
        #    pre-baked building tensor. NOT shapely — that is offline-only,
        #    see scripts/prep_osm.py. Also gates the sensor: observation needs
        #    an unoccluded ray, which is an ANGLE constraint (clear the
        #    roofline), not a radius — see AGENTS.md.
        # 5. Per-link path loss by class (A2A: FSPL + blockage; A2G: TR 36.777
        #    UMi-AV) -> channel.received_power_dbm -> channel.sinr_db. Pass a
        #    tx_mask holding only the transmitters active in the evaluated
        #    slot — it carries the MAC assumption -> channel.capacity_mbps
        # 6. Mission link: routing.best_relay_capacity over the drones that
        #    currently hold a valid HVT observation -> min_i(C_i)/min(n,3)
        #    NOTE: the channel fidelity level (F0 radius / F1 +occlusion /
        #    F2 +jammer / F3 full) is a config flag gating steps 4-6. RQ1.
        # 7. is_link_alive = routing.link_alive(C_e2e, CAPACITY_THRESHOLD_MBPS)
        # 8. Continuous GNN edge weights: sigmoid((capacity - 5.0) * gamma)
        #    (used by the model, not the env — env just exposes capacities)
        # 9. Reward: tracking quality + capacity term - energy penalty
        #    - lambda * Var(battery across agents)
        # 10. Termination: link down > MAX_JAM_STEPS consecutive, or any
        #     battery == 0

        observations = {a: self._build_observation(i) for i, a in enumerate(self.agents)}
        rewards = {a: 0.0 for a in self.agents}  # TODO
        terminations = {a: False for a in self.agents}  # TODO
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        self._step_count += 1
        return observations, rewards, terminations, truncations, infos

    def _build_observation(self, agent_idx: int) -> np.ndarray:
        # TODO: assemble the 13-dim vector — local kinematics, battery,
        # target tracking metrics (distance/FoV/occlusion to HVT), ambient
        # noise floor. Keep this agent-local (decentralized execution —
        # don't leak global state here; that's the critic's job).
        return np.zeros(OBS_DIM, dtype=np.float32)

    def render(self):
        # Deliberately not implemented here — see render.py. Training never
        # calls this; only periodic eval passes for wandb video logging.
        raise NotImplementedError("Use render.py for visualization")

    def close(self):
        pass
