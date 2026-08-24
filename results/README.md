# Results

Measured numbers, versioned with the code that produced them. Small, text, and
**tracked on purpose** — unlike `runs/`, which holds checkpoints and logs and is
gitignored because it is regenerable.

The practical reason: a sweep runs on a rented GPU pod and is analysed on a
laptop. Committing the summary means the pod pushes and the laptop pulls, rather
than scp-ing files between machines and losing track of which numbers came from
which commit.

| file | produced by | read by |
|---|---|---|
| `sweep_summary.jsonl` | `scripts/sweep.py` | `scripts/sweep.py --report-only` |

⚠️ Every row carries its own provenance — architecture, cadence, shaping, seed,
split, `env_steps`. It does **not** carry the device, and a device (and a CPU
architecture) is part of a measurement's provenance in this project
([`../docs/DECISIONS.md`](../docs/DECISIONS.md)). Sweep on one device.
