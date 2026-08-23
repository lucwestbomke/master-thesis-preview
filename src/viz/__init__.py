"""Drawing -- Block E.

Two front ends, one set of primitives:

    scripts/view_episode.py     INSPECTION. Whole-box map, source-polygon
                                overlay, `--worst`, the clearance timeline.
                                Its job is finding geometry bugs by looking,
                                which is how every geometry bug in this project
                                has actually been found.
    scripts/render_episode.py   PRESENTATION. Eval videos for wandb and vector
                                figures for the thesis, for any policy.

`scene.py` holds what both need -- oriented-box corners, the road graph, the
colour scheme. It was extracted from `view_episode.py` rather than copied:
forking would give two copies of the geometry drawing, and the whole reason that
code is trusted is that it draws **the boxes the env consumes**, not the source
polygons. Two copies drift, and the one that drifts is the one that stops
finding bugs.

Matplotlib only. It is already a dependency, and `AGENTS.md` requires flagging
new heavy ones.
"""

from .scene import (
    MIDRISE_M,
    TOWER_M,
    box_corners,
    draw_static_scene,
    inside_any_box,
    load_artefact,
    source_footprints,
)

__all__ = [
    "MIDRISE_M",
    "TOWER_M",
    "box_corners",
    "draw_static_scene",
    "inside_any_box",
    "load_artefact",
    "source_footprints",
]
