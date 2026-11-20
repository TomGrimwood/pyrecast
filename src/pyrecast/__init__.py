"""Recast (recastnavigation) navmeshes from Python.

    import pyrecast
    ground = pyrecast.Mesh(vertices, triangles)                    # (V, 3) float, (T, 3) int
    trees = pyrecast.Mesh(tree_vertices, tree_triangles, transforms=placings)   # (K, 4, 4) copies
    navmesh = pyrecast.build([ground, trees], pyrecast.Config(agent_radius=0.3), up="z")
    navmesh.vertices, navmesh.triangles      # the detail mesh: the walkable surface
    navmesh.polygon                          # each triangle's polygon
    navmesh.neighbours, navmesh.region       # each polygon's neighbours (-1: none) and region
    navmesh.cells.position, .neighbours      # the walkable cells: each one's floor, its 4 neighbours

The build is Recast's own (its demo's solo mesh): the meshes rasterized into a heightfield,
low obstacles, ledges and low ceilings filtered, the walkable area eroded by the agent's
radius, regions (watershed), contours, polygons and the detail mesh. Every setting is in
world units (`Config`). Recast is y up; `up="z"` turns a z-up world into it and back.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from . import _core

__all__ = ["Cells", "Config", "Mesh", "NavMesh", "build"]


@dataclass(frozen=True)
class Config:
    """How a navmesh is built, in world units and degrees; the defaults are Recast's demo's.

    `cell_size`, `cell_height`: the heightfield's voxels. `agent_height`: headroom needed;
    `agent_radius`: kept from walls and edges; `agent_max_climb`: highest step walked up;
    `agent_max_slope`: steepest floor. `region_min_area`: smaller islands are dropped;
    `region_merge_area`: smaller regions are joined to a neighbour (both areas, not cells).
    `edge_max_len`, `edge_max_error` (in cells): how outlines are simplified;
    `verts_per_poly`: 3 to 6. `detail_sample_dist`, `detail_sample_max_error`: how closely the
    detail mesh follows the surface (0 distance: polygons only). `two_sided`: a face is floor
    by its slope whichever way it winds (mirrored and left-handed meshes' floors face down);
    False is Recast's own test, its normal up. `drape`: the detail mesh's corners moved down
    onto the floor geometry beneath them (Recast's surface is its voxels' tops: up to two cell
    heights above the floor, and higher on slopes); with `detail_sample_dist` near the cell
    size, the surface follows the floor within a centimetre or two. Not Recast's own."""

    cell_size: float = 0.3
    cell_height: float = 0.2
    agent_height: float = 2.0
    agent_radius: float = 0.6
    agent_max_climb: float = 0.9
    agent_max_slope: float = 45.0
    region_min_area: float = 5.76
    region_merge_area: float = 36.0
    edge_max_len: float = 12.0
    edge_max_error: float = 1.3
    verts_per_poly: int = 6
    detail_sample_dist: float = 1.8
    detail_sample_max_error: float = 0.2
    two_sided: bool = True
    drape: bool = False


@dataclass(frozen=True)
class Mesh:
    """A triangle mesh: `vertices` (V, 3), `triangles` (T, 3) into them, and `transforms`
    (K, 4, 4), each a copy of it placed in the world (row vectors, p' = p M: translation in
    the last row); one copy where it is when None."""

    vertices: np.ndarray
    triangles: np.ndarray
    transforms: np.ndarray | None = field(default=None)


@dataclass(frozen=True)
class Cells:
    """The walkable cells the navmesh was made from (Recast's compact heightfield after the
    filters, the erosion and the small regions dropped): each one's `position` (S, 3), the
    middle of its floor (on the geometry's own surface with `Config.drape`); its `region`
    (S,); and its `neighbours` (S, 4), the cells it joins four ways: +x, -x, then +h and -h,
    h being y when z is up, else z (-1: none)."""

    position: np.ndarray
    region: np.ndarray
    neighbours: np.ndarray


@dataclass(frozen=True)
class NavMesh:
    """A built navmesh: the detail mesh's `vertices` (N, 3) and `triangles` (M, 3), each
    triangle's `polygon` (M,), each polygon's `neighbours` (P, verts_per_poly; -1 where an
    edge has none) and `region` (P,); and the walkable `cells` it was made from."""

    vertices: np.ndarray
    triangles: np.ndarray
    polygon: np.ndarray
    neighbours: np.ndarray
    region: np.ndarray
    cells: Cells


def build(
    meshes: list[Mesh],
    config: Config | None = None,
    bounds: tuple[ArrayLike, ArrayLike] | None = None,
    up: str = "y",
) -> NavMesh:
    """The navmesh over `meshes` for `config`, within `bounds` (lowest and highest corners;
    else round every copy of every mesh), in a world whose `up` is "y" (Recast's) or "z".
    Raises ValueError for bad input, RuntimeError when a step of Recast's fails."""
    config = config or Config()
    if up not in ("y", "z"):
        raise ValueError(f'up is "y" or "z", not {up!r}.')
    parts = []
    for mesh in meshes:
        v = np.ascontiguousarray(mesh.vertices, np.float32).reshape(-1, 3)
        t = np.ascontiguousarray(mesh.triangles, np.int32).reshape(-1, 3)
        m = np.eye(4)[None] if mesh.transforms is None else mesh.transforms
        m = np.ascontiguousarray(m, np.float64).reshape(-1, 4, 4)
        if len(v) and len(t) and len(m):
            parts.append((v, t, m))
    if not parts:
        none = Cells(
            np.zeros((0, 3), np.float32), np.zeros(0, np.int32), np.zeros((0, 4), np.int32)
        )
        return NavMesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32),
                       np.zeros(0, np.int32), np.zeros((0, config.verts_per_poly), np.int32),
                       np.zeros(0, np.int32), none)  # fmt: skip
    low, high = bounds if bounds is not None else _around(parts)
    low = np.asarray(low, np.float64).reshape(3).tolist()
    high = np.asarray(high, np.float64).reshape(3).tolist()
    settings = asdict(config)
    two_sided = settings.pop("two_sided")
    try:
        made = _core.build(
            parts, low, high,
            z_up=up == "z", two_sided=two_sided, **settings,
        )  # fmt: skip
    except (IndexError, OverflowError) as e:  # a triangle naming a vertex its mesh has not
        raise ValueError(str(e)) from e
    return NavMesh(*made[:5], Cells(*made[5:]))


def _around(parts) -> tuple[np.ndarray, np.ndarray]:
    """The box round every copy of every mesh."""
    lows, highs = [], []
    for v, _, m in parts:
        placed = np.einsum("vj,kji->kvi", v.astype(np.float64), m[:, :3, :3]) + m[:, None, 3, :3]
        lows.append(placed.reshape(-1, 3).min(axis=0))
        highs.append(placed.reshape(-1, 3).max(axis=0))
    return np.min(lows, axis=0), np.max(highs, axis=0)
