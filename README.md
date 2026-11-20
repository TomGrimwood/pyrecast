# pyrecast

[Recast](https://github.com/recastnavigation/recastnavigation)'s navmesh build from Python:
triangle meshes in (instanced copies too), a navmesh out as NumPy arrays. Recast is the
navmesh builder of Unity, Unreal, Godot and O3DE; this builds its 1.6.0 release, unaltered,
and calls it the way its own demo does.

```python
import numpy as np
import pyrecast

ground = pyrecast.Mesh(vertices, triangles)  # (V, 3) floats, (T, 3) ints
trees = pyrecast.Mesh(tree_vertices, tree_triangles, transforms=placings)  # (K, 4, 4): K copies
navmesh = pyrecast.build(
    [ground, trees],
    pyrecast.Config(cell_size=0.2, cell_height=0.02, agent_radius=0.25, agent_max_climb=0.35),
    bounds=((x0, y0, z0), (x1, y1, z1)),
    up="z",
)
navmesh.vertices, navmesh.triangles  # the detail mesh: the walkable surface
navmesh.polygon  # each triangle's polygon
navmesh.neighbours, navmesh.region  # each polygon's neighbours (-1: none) and region
navmesh.cells.position  # every walkable cell's floor
navmesh.cells.neighbours  # its neighbours, +x, -x, +h, -h (-1: none)
```

## What it does

Recast's solo-mesh build, each step Recast's own function: the meshes rasterized into a
heightfield, low obstacles, ledges and low ceilings filtered, the walkable area eroded by the
agent's radius, regions (watershed), contours, polygons and the detail mesh.

Added round it, nothing changed in it:

- **Instanced meshes**: a mesh with many `transforms` is passed once and placed copy by copy,
  never all at once: a forest of a few tree prototypes is read once, not as 50 million
  triangles.
- **Up axis**: Recast is y up; `up="z"` turns a z-up world into it and back.
- **Two-sided floors** (`Config.two_sided`, on): a face is floor by its slope whichever way
  it winds; a mirrored or left-handed mesh's floor faces down. Off is Recast's own test.
- **The walkable cells** (`NavMesh.cells`): the compact heightfield Recast built the polygons
  from, each cell's floor and its four neighbours: a dense surface, as some navigation needs.
- **Draping** (`Config.drape`, off): Recast's heights are its voxels' tops, up to two cell
  heights above the floor and higher on slopes (the highest point of the floor across a
  cell). With it on, the cells and the detail mesh's corners are moved down onto the floor
  geometry's own surface beneath them.

Every setting is in world units and degrees (`Config`), the defaults Recast's demo's.

## Building

`pip install .` or `uv sync`: CMake fetches Recast 1.6.0 (pinned by hash), nanobind binds it,
scikit-build-core packages it. Needs CMake 3.18 and a C++17 compiler (GCC, Clang or MSVC). The
wheel holds the `pyrecast` package and its extension only, Recast linked into it statically.

Wheels for Linux (x86_64, aarch64) and Windows (AMD64), CPython 3.12 and later, are built and
tested by cibuildwheel (`.github/workflows/wheels.yml`, settings in `pyproject.toml`): kept as
the run's artifacts, and attached to the GitHub release of a `v*` tag.

## Checks

```bash
uv sync
make        # lint, format, types, the tests
uv run python scripts/check.py   # what `make` runs, without make (Windows)
```

## Licence

zlib, as Recast's; built packages contain Recast, under its own notice (`LICENSE`).
