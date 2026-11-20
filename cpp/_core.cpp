// pyrecast._core: Recast's navmesh build (recastnavigation), callable from Python.
//
// The build is Recast's own solo-mesh pipeline, as its demo runs it (Sample_SoloMesh.cpp):
// rasterize, filter, compact, erode, regions (watershed), contours, polygons, detail. What
// is added here: meshes placed by many transforms (instanced copies, transformed one at a
// time, never all at once), a z-up world turned into Recast's y-up and back, walkability
// optionally either way up, and the result as NumPy arrays.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <Recast.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;
using Vertices = nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Triangles = nb::ndarray<const int, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Transforms = nb::ndarray<const double, nb::shape<-1, 4, 4>, nb::c_contig, nb::device::cpu>;
using Mesh = std::tuple<Vertices, Triangles, Transforms>;

// Recast's polygons have at most this many corners (Detour's limit too).
static const int MAX_VERTS_PER_POLY = 6;

template <typename T>
static nb::ndarray<nb::numpy, T> to_numpy(std::vector<T>&& data, std::vector<size_t> shape) {
    auto* owned = new std::vector<T>(std::move(data));
    nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return nb::ndarray<nb::numpy, T>(owned->data(), shape.size(), shape.data(), owner);
}

struct Free {
    void operator()(rcHeightfield* p) const { rcFreeHeightField(p); }
    void operator()(rcCompactHeightfield* p) const { rcFreeCompactHeightfield(p); }
    void operator()(rcContourSet* p) const { rcFreeContourSet(p); }
    void operator()(rcPolyMesh* p) const { rcFreePolyMesh(p); }
    void operator()(rcPolyMeshDetail* p) const { rcFreePolyMeshDetail(p); }
};

// Into Recast's y-up space from the caller's: z up (x, y, z) -> (x, z, -y), a turn.
static inline void in(const float* p, bool z_up, float* r) {
    if (z_up) { r[0] = p[0]; r[1] = p[2]; r[2] = -p[1]; }
    else { r[0] = p[0]; r[1] = p[1]; r[2] = p[2]; }
}
static inline void out(const float* r, bool z_up, float* p) {
    if (z_up) { p[0] = r[0]; p[1] = -r[2]; p[2] = r[1]; }
    else { p[0] = r[0]; p[1] = r[1]; p[2] = r[2]; }
}

// The floor's own surface, for draping Recast's onto it: for each cell centre a floor
// triangle covers, that triangle's plane (y = a x + b z + c, Recast's y up). Recast's heights
// are its voxels' tops: rounded up, and the highest point of the floor across a cell.
struct Planes {
    struct Entry { int cell; float a, b, c; };
    std::vector<Entry> entries;
    std::vector<size_t> first;  // each cell's first entry, after `finish`
    float origin[2] = {0, 0}, cs = 1;
    int width = 0, height = 0;

    void start(const rcConfig& cfg) {
        origin[0] = cfg.bmin[0]; origin[1] = cfg.bmin[2]; cs = cfg.cs;
        width = cfg.width; height = cfg.height;
    }
    void keep(const float* p0, const float* p1, const float* p2) {
        // The plane through the triangle, over x and z.
        const float d1x = p1[0] - p0[0], d1z = p1[2] - p0[2], d2x = p2[0] - p0[0], d2z = p2[2] - p0[2];
        const float det = d1x * d2z - d2x * d1z;
        if (std::fabs(det) < 1e-12f)
            return;
        const float d1y = p1[1] - p0[1], d2y = p2[1] - p0[1];
        const float a = (d1y * d2z - d2y * d1z) / det, b = (d2y * d1x - d1y * d2x) / det;
        const float c = p0[1] - a * p0[0] - b * p0[2];
        const float lox = std::min({p0[0], p1[0], p2[0]}), hix = std::max({p0[0], p1[0], p2[0]});
        const float loz = std::min({p0[2], p1[2], p2[2]}), hiz = std::max({p0[2], p1[2], p2[2]});
        const int x0 = std::max(0, (int)std::ceil((lox - origin[0]) / cs - 0.5f));
        const int x1 = std::min(width - 1, (int)std::floor((hix - origin[0]) / cs - 0.5f));
        const int z0 = std::max(0, (int)std::ceil((loz - origin[1]) / cs - 0.5f));
        const int z1 = std::min(height - 1, (int)std::floor((hiz - origin[1]) / cs - 0.5f));
        for (int x = x0; x <= x1; ++x)
            for (int z = z0; z <= z1; ++z) {
                const float px = origin[0] + (x + 0.5f) * cs, pz = origin[1] + (z + 0.5f) * cs;
                // Inside the triangle, seen from above.
                const float u = ((px - p0[0]) * d2z - d2x * (pz - p0[2])) / det;
                const float v = (d1x * (pz - p0[2]) - (px - p0[0]) * d1z) / det;
                if (u >= -1e-5f && v >= -1e-5f && u + v <= 1 + 1e-5f)
                    entries.push_back({x + z * width, a, b, c});
            }
    }
    void finish() {
        std::sort(entries.begin(), entries.end(), [](const Entry& l, const Entry& r) { return l.cell < r.cell; });
        first.assign((size_t)width * height + 1, entries.size());
        for (size_t i = entries.size(); i-- > 0;)
            first[entries[i].cell] = i;
        for (size_t c = first.size() - 1; c-- > 0;)
            first[c] = std::min(first[c], first[c + 1]);
    }
    // Moves `v` down onto the nearest plane below it, within `reach`, of its own cell; of the
    // cells round it only when its own has none (a plane carried across a neighbour's cell,
    // on ground of small triangles, can be anywhere). Leaves it where it is when none fits.
    void drape(float* v, float reach) const {
        const int cx = (int)std::floor((v[0] - origin[0]) / cs), cz = (int)std::floor((v[2] - origin[1]) / cs);
        for (int ring = 0; ring <= 1; ++ring) {
            float best = 1e30f;
            for (int dz = -ring; dz <= ring; ++dz)
                for (int dx = -ring; dx <= ring; ++dx) {
                    if (ring && !dx && !dz)
                        continue;
                    const int x = cx + dx, z = cz + dz;
                    if (x < 0 || z < 0 || x >= width || z >= height)
                        continue;
                    const int cell = x + z * width;
                    for (size_t i = first[cell]; i < first[cell + 1]; ++i) {
                        const Entry& e = entries[i];
                        const float below = v[1] - (e.a * v[0] + e.b * v[2] + e.c);
                        if (below >= -0.01f && below <= reach && below < best)
                            best = below;
                    }
                }
            if (best < 1e30f) {
                v[1] -= best;
                return;
            }
        }
    }
};

static nb::tuple build(const std::vector<Mesh>& meshes, std::vector<float> bounds_min,
                       std::vector<float> bounds_max, bool z_up, bool two_sided,
                       float cell_size, float cell_height, float agent_height,
                       float agent_radius, float agent_max_climb, float agent_max_slope,
                       float region_min_area, float region_merge_area, float edge_max_len,
                       float edge_max_error, int verts_per_poly, float detail_sample_dist,
                       float detail_sample_max_error, bool drape) {
    if (bounds_min.size() != 3 || bounds_max.size() != 3)
        throw std::invalid_argument("bounds are three numbers each");
    if (verts_per_poly < 3 || verts_per_poly > MAX_VERTS_PER_POLY)
        throw std::invalid_argument("verts_per_poly is 3 to 6");
    rcConfig cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.cs = cell_size;
    cfg.ch = cell_height;
    cfg.walkableSlopeAngle = agent_max_slope;
    cfg.walkableHeight = (int)std::ceil(agent_height / cell_height);
    cfg.walkableClimb = (int)std::floor(agent_max_climb / cell_height);
    cfg.walkableRadius = (int)std::ceil(agent_radius / cell_size);
    cfg.maxEdgeLen = (int)(edge_max_len / cell_size);
    cfg.maxSimplificationError = edge_max_error;
    cfg.minRegionArea = (int)(region_min_area / (cell_size * cell_size));
    cfg.mergeRegionArea = (int)(region_merge_area / (cell_size * cell_size));
    cfg.maxVertsPerPoly = verts_per_poly;
    cfg.detailSampleDist = detail_sample_dist < 0.9f * cell_size ? 0 : detail_sample_dist;
    cfg.detailSampleMaxError = detail_sample_max_error;
    float a[3], b[3];
    in(bounds_min.data(), z_up, a);
    in(bounds_max.data(), z_up, b);
    for (int i = 0; i < 3; ++i) { cfg.bmin[i] = std::min(a[i], b[i]); cfg.bmax[i] = std::max(a[i], b[i]); }
    rcCalcGridSize(cfg.bmin, cfg.bmax, cfg.cs, &cfg.width, &cfg.height);
    if (cfg.width <= 0 || cfg.height <= 0)
        throw std::invalid_argument("the bounds are empty");
    for (const auto& [vertices, triangles, transforms] : meshes) {
        const int* t = triangles.data();
        for (size_t i = 0; i < 3 * triangles.shape(0); ++i)
            if (t[i] < 0 || (size_t)t[i] >= vertices.shape(0))
                throw std::out_of_range("a triangle names a vertex its mesh has not");
    }

    rcContext ctx(false);
    std::unique_ptr<rcHeightfield, Free> hf(rcAllocHeightfield());
    std::unique_ptr<rcCompactHeightfield, Free> chf(rcAllocCompactHeightfield());
    std::unique_ptr<rcContourSet, Free> cset(rcAllocContourSet());
    std::unique_ptr<rcPolyMesh, Free> pmesh(rcAllocPolyMesh());
    std::unique_ptr<rcPolyMeshDetail, Free> dmesh(rcAllocPolyMeshDetail());
    const float walkable = std::cos(agent_max_slope * 3.14159265358979f / 180.0f);
    Planes planes;
    const char* failed = nullptr;
    {
        nb::gil_scoped_release released;
        if (!rcCreateHeightfield(&ctx, *hf, cfg.width, cfg.height, cfg.bmin, cfg.bmax, cfg.cs, cfg.ch))
            failed = "Recast could not make the heightfield";
        std::vector<float> placed;
        std::vector<unsigned char> areas;
        planes.start(cfg);
        for (size_t mi = 0; !failed && mi < meshes.size(); ++mi) {
            const auto& [vertices, triangles, transforms] = meshes[mi];
            const size_t nv = vertices.shape(0), nt = triangles.shape(0);
            const float* v = vertices.data();
            const int* t = triangles.data();
            placed.resize(nv * 3);
            areas.resize(nt);
            for (size_t k = 0; !failed && k < transforms.shape(0); ++k) {
                const double* m = transforms.data() + 16 * k;  // row vectors: p' = p M
                float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
                for (size_t i = 0; i < nv; ++i) {
                    const double x = v[3 * i], y = v[3 * i + 1], z = v[3 * i + 2];
                    const float w[3] = {(float)(x * m[0] + y * m[4] + z * m[8] + m[12]),
                                        (float)(x * m[1] + y * m[5] + z * m[9] + m[13]),
                                        (float)(x * m[2] + y * m[6] + z * m[10] + m[14])};
                    float* r = &placed[3 * i];
                    in(w, z_up, r);
                    for (int j = 0; j < 3; ++j) { lo[j] = std::min(lo[j], r[j]); hi[j] = std::max(hi[j], r[j]); }
                }
                bool away = false;
                for (int j = 0; j < 3; ++j)
                    away |= hi[j] < cfg.bmin[j] - cfg.cs || lo[j] > cfg.bmax[j] + cfg.cs;
                if (away)
                    continue;
                for (size_t i = 0; i < nt; ++i) {
                    const float* p0 = &placed[3 * t[3 * i]];
                    const float* p1 = &placed[3 * t[3 * i + 1]];
                    const float* p2 = &placed[3 * t[3 * i + 2]];
                    const float e0[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
                    const float e1[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
                    const float n[3] = {e0[1] * e1[2] - e0[2] * e1[1], e0[2] * e1[0] - e0[0] * e1[2],
                                        e0[0] * e1[1] - e0[1] * e1[0]};
                    const float size = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                    const float up = two_sided ? std::fabs(n[1]) : n[1];
                    areas[i] = (size > 0 && up >= walkable * size) ? RC_WALKABLE_AREA : RC_NULL_AREA;
                }
                if (!rcRasterizeTriangles(&ctx, placed.data(), (int)nv, t, areas.data(), (int)nt,
                                          *hf, cfg.walkableClimb))
                    failed = "Recast could not rasterize a mesh";
                if (drape)
                    for (size_t i = 0; i < nt; ++i)
                        if (areas[i] == RC_WALKABLE_AREA)
                            planes.keep(&placed[3 * t[3 * i]], &placed[3 * t[3 * i + 1]],
                                        &placed[3 * t[3 * i + 2]]);
            }
        }
        if (!failed) {
            rcFilterLowHangingWalkableObstacles(&ctx, cfg.walkableClimb, *hf);
            rcFilterLedgeSpans(&ctx, cfg.walkableHeight, cfg.walkableClimb, *hf);
            rcFilterWalkableLowHeightSpans(&ctx, cfg.walkableHeight, *hf);
        }
        if (!failed && !rcBuildCompactHeightfield(&ctx, cfg.walkableHeight, cfg.walkableClimb, *hf, *chf))
            failed = "Recast could not compact the heightfield";
        hf.reset();
        if (!failed && !rcErodeWalkableArea(&ctx, cfg.walkableRadius, *chf))
            failed = "Recast could not erode the walkable area";
        if (!failed && !rcBuildDistanceField(&ctx, *chf))
            failed = "Recast could not build the distance field";
        if (!failed && !rcBuildRegions(&ctx, *chf, 0, cfg.minRegionArea, cfg.mergeRegionArea))
            failed = "Recast could not build regions";
        if (!failed && !rcBuildContours(&ctx, *chf, cfg.maxSimplificationError, cfg.maxEdgeLen, *cset))
            failed = "Recast could not build contours";
        if (!failed && !rcBuildPolyMesh(&ctx, *cset, cfg.maxVertsPerPoly, *pmesh))
            failed = "Recast could not build polygons: more than 65,535 corners (build smaller bounds)";
        if (!failed && !rcBuildPolyMeshDetail(&ctx, *pmesh, *chf, cfg.detailSampleDist,
                                              cfg.detailSampleMaxError, *dmesh))
            failed = "Recast could not build the detail mesh";
    }
    if (failed)
        throw std::runtime_error(failed);
    if (drape) {
        // How far above the geometry Recast's surface can be: its rounding up (two cell
        // heights) and a cell's rise at the steepest floor.
        const float reach = 2.0f * cfg.ch + cfg.cs * std::tan(agent_max_slope * 3.14159265358979f / 180.0f) + 0.01f;
        planes.finish();
        for (int i = 0; i < dmesh->nverts; ++i)
            planes.drape(&dmesh->verts[3 * i], reach);
    }

    std::vector<float> vertices((size_t)dmesh->nverts * 3);
    for (int i = 0; i < dmesh->nverts; ++i)
        out(&dmesh->verts[3 * i], z_up, &vertices[3 * i]);
    std::vector<int> triangles, polygon;
    triangles.reserve((size_t)dmesh->ntris * 3);
    polygon.reserve(dmesh->ntris);
    for (int p = 0; p < dmesh->nmeshes; ++p) {
        const unsigned int* sub = &dmesh->meshes[4 * p];
        for (unsigned int j = 0; j < sub[3]; ++j) {
            const unsigned char* tri = &dmesh->tris[4 * (sub[2] + j)];
            const int i0 = (int)(sub[0] + tri[0]), i1 = (int)(sub[0] + tri[1]), i2 = (int)(sub[0] + tri[2]);
            // Recast's winding, the other way round when z is up (the turn mirrors its order).
            if (z_up) triangles.insert(triangles.end(), {i0, i2, i1});
            else triangles.insert(triangles.end(), {i0, i1, i2});
            polygon.push_back(p);
        }
    }
    const int nvp = pmesh->nvp;
    std::vector<int> neighbours((size_t)pmesh->npolys * nvp, -1), region(pmesh->npolys);
    for (int p = 0; p < pmesh->npolys; ++p) {
        region[p] = pmesh->regs[p];
        for (int j = 0; j < nvp; ++j) {
            const unsigned short n = pmesh->polys[p * 2 * nvp + nvp + j];
            if (n != RC_MESH_NULL_IDX && !(n & 0x8000))
                neighbours[(size_t)nvp * p + j] = n;
        }
    }
    // The walkable cells: every span of the compact heightfield left in a region (after the
    // filters, the erosion and the small regions dropped), its middle on its floor (draped:
    // on the geometry's own surface there), its region, and its neighbours four ways in the
    // caller's axes: +x, -x, +h, -h (h: y when z is up, else z).
    std::vector<int> index((size_t)chf->spanCount, -1);
    std::vector<float> cell_position;
    std::vector<int> cell_region;
    const float reach = 2.0f * cfg.ch + cfg.cs * std::tan(agent_max_slope * 3.14159265358979f / 180.0f) + 0.01f;
    int kept = 0;
    for (int z = 0; z < chf->height; ++z)
        for (int x = 0; x < chf->width; ++x) {
            const rcCompactCell& c = chf->cells[x + z * chf->width];
            for (int i = (int)c.index; i < (int)(c.index + c.count); ++i) {
                const rcCompactSpan& sp = chf->spans[i];
                if (chf->areas[i] == RC_NULL_AREA || sp.reg == 0)
                    continue;
                float r[3] = {chf->bmin[0] + (x + 0.5f) * cfg.cs, chf->bmin[1] + (sp.y + 1) * cfg.ch,
                              chf->bmin[2] + (z + 0.5f) * cfg.cs};
                if (drape)
                    planes.drape(r, reach);
                float w[3];
                out(r, z_up, w);
                cell_position.insert(cell_position.end(), {w[0], w[1], w[2]});
                cell_region.push_back(sp.reg & ~RC_BORDER_REG);
                index[i] = kept++;
            }
        }
    // Recast's directions (0: -x, 1: +z, 2: +x, 3: -z) as +x, -x, +h, -h in the caller's axes.
    const int way_of[4] = {1, z_up ? 3 : 2, 0, z_up ? 2 : 3};
    std::vector<int> cell_neighbours((size_t)kept * 4, -1);
    for (int z = 0; z < chf->height; ++z)
        for (int x = 0; x < chf->width; ++x) {
            const rcCompactCell& c = chf->cells[x + z * chf->width];
            for (int i = (int)c.index; i < (int)(c.index + c.count); ++i) {
                if (index[i] < 0)
                    continue;
                const rcCompactSpan& sp = chf->spans[i];
                for (int dir = 0; dir < 4; ++dir) {
                    const int con = rcGetCon(sp, dir);
                    if (con == RC_NOT_CONNECTED)
                        continue;
                    const int ax = x + rcGetDirOffsetX(dir), az = z + rcGetDirOffsetY(dir);
                    const int j = (int)chf->cells[ax + az * chf->width].index + con;
                    cell_neighbours[(size_t)4 * index[i] + way_of[dir]] = index[j];
                }
            }
        }
    const size_t nverts = dmesh->nverts, ntris = polygon.size(), npolys = pmesh->npolys;
    return nb::make_tuple(to_numpy(std::move(vertices), {nverts, 3}),
                          to_numpy(std::move(triangles), {ntris, 3}),
                          to_numpy(std::move(polygon), {ntris}),
                          to_numpy(std::move(neighbours), {npolys, (size_t)nvp}),
                          to_numpy(std::move(region), {npolys}),
                          to_numpy(std::move(cell_position), {(size_t)kept, 3}),
                          to_numpy(std::move(cell_region), {(size_t)kept}),
                          to_numpy(std::move(cell_neighbours), {(size_t)kept, 4}));
}

NB_MODULE(_core, m) {
    m.doc() = "Recast's navmesh build (recastnavigation): pyrecast.build calls it.";
    m.def("build", &build, nb::arg("meshes"), nb::arg("bounds_min"), nb::arg("bounds_max"),
          nb::arg("z_up"), nb::arg("two_sided"), nb::arg("cell_size"), nb::arg("cell_height"),
          nb::arg("agent_height"), nb::arg("agent_radius"), nb::arg("agent_max_climb"),
          nb::arg("agent_max_slope"), nb::arg("region_min_area"), nb::arg("region_merge_area"),
          nb::arg("edge_max_len"), nb::arg("edge_max_error"), nb::arg("verts_per_poly"),
          nb::arg("detail_sample_dist"), nb::arg("detail_sample_max_error"), nb::arg("drape"));
}
