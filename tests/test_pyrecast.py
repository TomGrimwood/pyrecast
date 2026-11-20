"""pyrecast on made-up scenes: where an agent can walk, and the interface's promises."""

import numpy as np
import pytest

import pyrecast

#: An agent as a person: 20 cm cells, kept 25 cm from edges, steps of 35 cm.
PERSON = pyrecast.Config(
    cell_size=0.2, cell_height=0.02, agent_height=1.8, agent_radius=0.25,
    agent_max_climb=0.35, agent_max_slope=40.0, region_min_area=1.0,
)  # fmt: skip


def quad(x0, y0, x1, y1, z) -> pyrecast.Mesh:
    """A flat rectangle at height `z` (z up)."""
    v = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], float)
    return pyrecast.Mesh(v, np.array([[0, 1, 2], [0, 2, 3]]))


def box(x0, y0, z0, x1, y1, z1) -> pyrecast.Mesh:
    """A closed box (z up)."""
    v = np.array([[x, y, z] for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)], float)
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return pyrecast.Mesh(v, np.array([[f[0], f[k], f[k + 1]] for f in faces for k in (1, 2)]))


def cells_near(navmesh, x, y, within=0.15):
    """The floors' heights over (x, y): of the cells whose middle is within `within` of it in
    plan, each height once."""
    p = navmesh.cells.position
    near = np.hypot(p[:, 0] - x, p[:, 1] - y) <= within
    return sorted(set(np.round(p[near, 2], 1).tolist()))


def pieces(navmesh) -> np.ndarray:
    """Each cell's piece: the cells joined through their neighbours."""
    n = navmesh.cells.neighbours
    piece = np.arange(len(n))
    changed = True
    while changed:
        joined = np.where(n >= 0, piece[np.maximum(n, 0)], piece[:, None]).min(axis=1)
        changed = bool((joined < piece).any())
        piece = np.minimum(piece, joined)
    return piece


def test_a_floor_is_walkable_but_its_edge_kept_an_agents_radius_away():
    made = pyrecast.build([quad(0, 0, 10, 10, 0)], PERSON, up="z")
    p = made.cells.position
    assert len(p) and np.allclose(p[:, 2], p[0, 2])
    # No nearer the edge than the radius (Recast also takes a cell more where the floor
    # meets its bounds, as at a drop), and not much further.
    assert p[:, :2].min() >= 0.25 and p[:, :2].max() <= 9.75
    assert p[:, :2].min() <= 0.25 + 3 * PERSON.cell_size
    assert len(made.triangles) and made.vertices[:, 2].min() >= 0.0


def test_what_stands_in_the_way_is_walked_round():
    made = pyrecast.build([quad(0, 0, 10, 10, 0), box(4.5, 4.5, 0, 5.5, 5.5, 2.0)], PERSON, up="z")
    assert cells_near(made, 5, 5) == []  # the pillar
    assert cells_near(made, 5, 5.65) == []  # nearer to it than the radius
    assert cells_near(made, 5, 6.1) != []
    assert len(set(pieces(made))) == 1  # one floor all round it


def test_a_step_is_climbed_and_a_ledge_is_not():
    floor = quad(0, 0, 20, 10, 0)
    made = pyrecast.build(
        [floor, box(3, 3, 0, 7, 7, 0.3), box(13, 3, 0, 17, 7, 0.8)], PERSON, up="z"
    )
    piece = pieces(made)
    p = made.cells.position

    def at(x, y):
        return piece[np.argmin(np.hypot(p[:, 0] - x, p[:, 1] - y))]

    assert cells_near(made, 5, 5)[0] == pytest.approx(0.3, abs=0.1)  # on the step
    assert at(5, 5) == at(1, 1)  # walked up onto it from the floor
    assert cells_near(made, 15, 5)[0] == pytest.approx(0.8, abs=0.1)  # on the ledge
    assert at(15, 5) != at(1, 1)  # but not reached from the floor


def test_floors_one_over_another_are_both_kept_with_headroom_and_not_without():
    made = pyrecast.build([quad(0, 0, 10, 10, 0), quad(2, 2, 8, 8, 3.0)], PERSON, up="z")
    assert len(cells_near(made, 5, 5)) == 2  # the floor below (3 m of headroom) and the deck
    low = pyrecast.build([quad(0, 0, 10, 10, 0), quad(2, 2, 8, 8, 1.2)], PERSON, up="z")
    assert len(cells_near(low, 5, 5)) == 1  # under a deck 1.2 m up: no room


def test_copies_of_a_mesh_are_as_the_same_mesh_placed_twice():
    pillar = box(0, 0, 0, 1, 1, 2)
    places = np.tile(np.eye(4), (2, 1, 1))
    places[0, 3, :3] = (3, 3, 0)
    places[1, 3, :3] = (6, 6, 0)
    copies = pyrecast.Mesh(pillar.vertices, pillar.triangles, transforms=places)
    one = [quad(0, 0, 10, 10, 0), copies]
    two = [quad(0, 0, 10, 10, 0), box(3, 3, 0, 4, 4, 2), box(6, 6, 0, 7, 7, 2)]
    a, b = pyrecast.build(one, PERSON, up="z"), pyrecast.build(two, PERSON, up="z")
    assert np.array_equal(a.cells.position, b.cells.position)


def test_a_y_up_world_is_the_z_up_one_turned():
    scene = [quad(0, 0, 10, 10, 0), box(4, 4, 0, 6, 6, 2)]
    z_up = pyrecast.build(scene, PERSON, up="z")
    turn = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])  # (x, y, z) z up -> (x, z, -y) y up
    y_up = pyrecast.build([pyrecast.Mesh(m.vertices @ turn, m.triangles) for m in scene], PERSON)
    back = y_up.cells.position @ turn.T
    order = lambda p: p[np.lexsort((p[:, 1], p[:, 0]))]  # noqa: E731
    assert np.allclose(order(back), order(z_up.cells.position), atol=1e-4)


def test_draped_cells_and_corners_lie_on_a_slope_recasts_own_above_it():
    v = np.array([[0, 0, 0], [20, 0, 2], [20, 20, 2], [0, 20, 0]], float)  # a 10% rise in x
    slope = pyrecast.Mesh(v, np.array([[0, 1, 2], [0, 2, 3]]))
    for drape in (False, True):
        config = pyrecast.Config(**{**PERSON.__dict__, "drape": drape})
        made = pyrecast.build([slope], config, up="z")
        above = made.cells.position[:, 2] - 0.1 * made.cells.position[:, 0]
        corners = made.vertices[:, 2] - 0.1 * made.vertices[:, 0]
        if drape:
            assert np.abs(above).max() < 1e-3 and np.abs(corners).max() < 1e-3
        else:
            assert above.min() > 0.01  # Recast's surface: its voxels' tops


def test_neighbours_join_both_ways():
    made = pyrecast.build([quad(0, 0, 10, 10, 0), box(4, 4, 0, 6, 6, 2)], PERSON, up="z")
    n = made.cells.neighbours
    opposite = (1, 0, 3, 2)  # +x, -x, +h, -h
    for way in range(4):
        has = n[:, way] >= 0
        assert np.array_equal(n[n[has, way], opposite[way]], np.flatnonzero(has))


def test_what_is_wrong_is_said():
    floor = quad(0, 0, 10, 10, 0)
    with pytest.raises(ValueError):
        pyrecast.build([pyrecast.Mesh(floor.vertices, np.array([[0, 1, 9]]))], PERSON, up="z")
    with pytest.raises(ValueError):
        pyrecast.build([floor], PERSON, up="x")
    with pytest.raises(ValueError):
        pyrecast.build([floor], pyrecast.Config(verts_per_poly=9))
    empty = pyrecast.build([], PERSON)
    assert len(empty.cells.position) == 0 and len(empty.triangles) == 0
