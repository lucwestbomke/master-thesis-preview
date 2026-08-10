"""
Environment package.

Built and unit-tested (see the co-located `test_*.py` for hand-computed values):

    channel.py   path loss by link class, SINR with intra-swarm interference,
                 Shannon rate with a modulation cap
    routing.py   hop-limited widest-path DP; end-to-end rate min(C_i)/min(n,3)
    energy.py    rotary-wing propulsion power (U-shaped), radio DC draw
    reward.py    pure-function reward: mission term + potential-based shaping

Not built yet:

    occlusion.py   batched segment-vs-box (slab method), 2.5D   -- Block C
    core.py        batched env with a leading num_envs dimension -- Block D

swarm_env.py is the PettingZoo *adapter*, for API-compliance tests and visual
debugging only. Training runs against the batched core; see AGENTS.md.
"""
