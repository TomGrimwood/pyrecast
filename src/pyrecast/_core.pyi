"""Recast's navmesh build (recastnavigation): pyrecast.build calls it."""

from collections.abc import Sequence
from typing import Annotated

import numpy
from numpy.typing import NDArray


def build(meshes: Sequence[tuple[Annotated[NDArray[numpy.float32], dict(shape=(None, 3), order='C', device='cpu', writable=False)], Annotated[NDArray[numpy.int32], dict(shape=(None, 3), order='C', device='cpu', writable=False)], Annotated[NDArray[numpy.float64], dict(shape=(None, 4, 4), order='C', device='cpu', writable=False)]]], bounds_min: Sequence[float], bounds_max: Sequence[float], z_up: bool, two_sided: bool, cell_size: float, cell_height: float, agent_height: float, agent_radius: float, agent_max_climb: float, agent_max_slope: float, region_min_area: float, region_merge_area: float, edge_max_len: float, edge_max_error: float, verts_per_poly: int, detail_sample_dist: float, detail_sample_max_error: float, drape: bool) -> tuple: ...
