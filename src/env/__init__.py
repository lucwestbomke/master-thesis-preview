"""
Environment package.

Built and unit-tested (see the co-located `test_*.py` for hand-computed values):

    channel.py   path loss by link class, SINR with intra-swarm interference,
                 Shannon rate with a modulation cap
    routing.py   hop-limited widest-path DP; end-to-end rate min(C_i)/min(n,3)
    energy.py    rotary-wing propulsion power (U-shaped), radio DC draw
    reward.py    pure-function reward: mission term + potential-based shaping

    occlusion.py batched segment-vs-oriented-box (slab method), 2.5D
    core.py      the batched env, leading num_envs dimension -- THE training path

swarm_env.py is the PettingZoo *adapter*, for API-compliance tests and visual
debugging only. Training runs against `core.BatchedSwarmEnv` through
`src/training/skrl_wrapper.py`; see AGENTS.md.
"""
