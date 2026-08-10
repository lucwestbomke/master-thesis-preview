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
AGENTS.md "Hard rules" and docs/DECISIONS.md.

Purely numeric/tensor state — no rendering happens here. See render.py for
the separate matplotlib top-down visualizer used for eval videos / figures.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import torch
from gymnasium import spaces
from pettingzoo import ParallelEnv

# Ego observation. Agent-local only -- global state belongs to the critic.
# Breakdown (see docs/ENVIRONMENT.md "Observations"):
#   3  own velocity          3  relative vector to HVT
#   1  own altitude          3  relative velocity of HVT
#   1  battery               3  relative vector to MCV
#   1  sees HVT (soft)       1  measured noise floor
#   1  clearance to HVT      1  clearance to MCV
#   1  on relay path         1  current e2e capacity
#   1  steps since link OK
OBS_DIM = 21
# Per-neighbour: rel pos (3), rel vel (3), battery, sees_hvt, on_path.
NEIGHBOUR_DIM = 9
# Per-edge: link capacity, ray clearance margin. The GNN's only extra input
# over DeepSets -- this is the rung RQ2 tests.
EDGE_DIM = 2
# Motion only. Transmit power is fixed at 30 dBm -- adaptive Ptx was tested
# against fair baselines under three separate justifications (energy,
# interference, detectability) and came out null each time. See
# docs/NEGATIVE_RESULTS.md before adding a 4th dimension back.
ACTION_DIM = 3  # dv_x, dv_y, dv_z
PTX_FIXED_DBM = 30.0
CAPACITY_THRESHOLD_MBPS = 5.0

# Mission failure is a PER-STEP condition, never a terminal event. Terminating
# on link loss breaks the task two ways: the policy learns to never acquire the
# HVT so it can never fail, and a random initial policy dies within a handful of
# steps and therefore never experiences the tracking phase at all. Episodes run
# to EPISODE_STEPS or until a battery dies -- battery death is physical and
# cannot be gamed, since hovering at the MCV burns power too.
EPISODE_STEPS = 600  # 240 s at DT_SECONDS
DT_SECONDS = 0.4

# Drones launch parked on the MCV. The HVT starts CLOSE (300-500 m) and drives
# away, so the chain requirement escalates 1 -> 2 -> 3 hops during the episode.
# This resolves an otherwise unsolvable conflict: the MCV must be far for a
# relay to be necessary at all (one drone covers everything inside ~1000 m) but
# near for the cue to survive transit.
HVT_START_RANGE_M = (300.0, 500.0)

# One-shot cue at launch, NEVER refreshed. Its job is to break directional
# symmetry and give early training a gradient -- not to solve acquisition.
# Precision barely matters; drift during transit swamps sigma. A refreshed cue
# would imply a persistent external tracker, which makes the swarm redundant.
CUE_SIGMA_M = 150.0

# Route is PRE-SAMPLED on the in-box road graph at reset, not chosen randomly at
# junctions -- that doubles back, stalls in cul-de-sacs and leaves the map.
# Pre-sampling gives the map border for free. Speeds come from OSM road class;
# primary/trunk are excluded so the drone keeps a 1.4-1.8x speed margin.
HVT_SPEED_BY_CLASS_MS = {"residential": 8.3, "secondary": 13.9}
DRONE_CRUISE_MS = 20.0
DRONE_DASH_MS = 25.0


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
        self._hvt_position: torch.Tensor | None = None  # (3,)
        self._hvt_route: torch.Tensor | None = None  # (T, 3) waypoints
        self._hvt_cue: torch.Tensor | None = None  # (3,) one-shot, never refreshed
        self._acquired: bool = False  # has the swarm seen the HVT directly yet
        self._step_count = 0

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

        # TODO: randomize MCV position on the map, then sample the HVT start on
        # a road HVT_START_RANGE_M from it.
        # TODO: PRE-SAMPLE the whole route as a path on the in-box road graph,
        # required to move outward from the MCV. Pre-sampling (not random turns
        # at junctions) keeps the HVT inside the map by construction, fixes the
        # episode duration, and is reproducible from the seed. Per-segment speed
        # from HVT_SPEED_BY_CLASS_MS; exclude primary/trunk so the drone keeps a
        # 1.4-1.8x speed margin.
        # TODO: place all drones ON the MCV (they launch from it), velocities
        # zero, battery 1.0. The chain forms during transit -- that is part of
        # the mission, not a preamble to it.
        # TODO: set self._hvt_cue once = true position + N(0, CUE_SIGMA_M).
        # Never refresh it. See docs/ENVIRONMENT.md for why.

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
        #    roofline), not a radius — see docs/PHYSICS.md.
        # 5. Per-link path loss by class (A2A: FSPL + blockage; A2G: TR 36.777
        #    UMi-AV) -> channel.received_power_dbm -> channel.sinr_db. Pass a
        #    tx_mask holding only the transmitters active in the evaluated
        #    slot — it carries the MAC assumption -> channel.capacity_mbps
        # 6. Mission link: routing.best_relay_capacity over the drones that
        #    currently hold a valid HVT observation -> min_i(C_i)/min(n,3)
        #    NOTE: the channel fidelity level gates steps 4-6 and is RQ1's
        #    independent variable. F0 radius / F1 +occlusion / F2 +SINR&rate /
        #    F3 +jammer / F4 +multi-hop division. It is a CONSTRUCTION-TIME
        #    config flag and must never change within a run -- unlike the
        #    curriculum, which varies within every run on an identical
        #    schedule across all conditions. See docs/ENVIRONMENT.md "Curriculum".
        # 7. is_link_alive = routing.link_alive(C_e2e, CAPACITY_THRESHOLD_MBPS)
        # 8. Continuous GNN edge weights: sigmoid((capacity - 5.0) * gamma)
        #    (used by the model, not the env — env just exposes capacities)
        # 9. Reward -- see docs/REWARD.md for the full derivation:
        #      w_mission * [observed AND C_e2e >= threshold]   <- IS the metric
        #      + gamma*Phi(s') - Phi(s)                        <- PBRS, see below
        #      - w_idle * [not observed]                       <- kills the lazy
        #        optimum, which survives fixed-length episodes because never
        #        acquiring also means never flying out, i.e. saving energy
        #      - w_energy * power   - lambda * Var(battery)   - w_effort * |a|^2
        #    Phi = k*(w_a*Phi_approach + w_o*Phi_observe + w_l*Phi_link), all
        #    TEAM quantities (nearest drone / best clearance / e2e capacity) --
        #    per-drone potentials pull all five onto the HVT and nobody relays.
        #    SUM not product: a product is flat at t=0 when both are ~0.
        #    Phi must be 0 at genuine terminal states or the invariance breaks.
        # 10. Termination: battery == 0 ONLY. Mission failure is a per-step
        #     condition recorded in `infos`, never terminal -- see the
        #     EPISODE_STEPS comment above for why. Truncate at EPISODE_STEPS.

        observations = {a: self._build_observation(i) for i, a in enumerate(self.agents)}
        rewards = {a: 0.0 for a in self.agents}  # TODO
        terminations = {a: False for a in self.agents}  # TODO: battery death only
        self._step_count += 1
        truncated = self._step_count >= EPISODE_STEPS
        truncations = {a: truncated for a in self.agents}
        # TODO: report per-step mission status here (observed / link_alive /
        # e2e capacity). "Fraction of steps mission-capable" is the primary
        # metric, and it cannot be gamed by refusing to start.
        infos = {a: {} for a in self.agents}

        return observations, rewards, terminations, truncations, infos

    def _build_observation(self, agent_idx: int) -> np.ndarray:
        # TODO: assemble the 21-dim ego vector — local kinematics, battery,
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
