# =============================================================================
# uvmap.em - UV unwrapping and texture-coordinate generation
#
#   planar_project        project along an axis
#   cylindrical_project   wrap around Y (seam-aware)
#   spherical_project     latitude/longitude (seam-aware)
#   box_project           per-face dominant-axis planar mapping
#   unweld                give every triangle corner its own vertex
#   atlas_pack_faces      pack each face into its own atlas cell (lightmap
#                         style: guaranteed overlap-free)
#   uv_bounds, scale_uvs, checker_test_pattern
#
# Seam-aware = triangles that straddle the wrap seam get duplicated vertices
# so the texture doesn't smear backwards across the whole surface.
# =============================================================================

import mathx;
import arrayx;
import vector;
import mesh;
import image;
import texture;
# emx-scope-safe

# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------

# Planar projection along +axis ("x"|"y"|"z"), fitted to the mesh bounds.
fn planar_project(m, axis) {
    int b = 0; int du = 0; int dv = 0; int hi = 0; int i = 0; int lo = 0; int nv = 0; int positions = 0; int ua = 0; int uvs = 0; int va = 0;
    b = mesh.bounds(m);
    lo = b[0];
    hi = b[1];
    positions = m[0];
    nv = len(positions) // 3;
    ua = 0; va = 1;                       # looking down z: u=x, v=y
    if (axis == "x") { ua = 2; va = 1; }
    elif (axis == "y") { ua = 0; va = 2; }
    du = mathx.maxv(hi[ua] - lo[ua], 1.0e-9);
    dv = mathx.maxv(hi[va] - lo[va], 1.0e-9);
    uvs = [];
    for (i = 0; i < nv; ++i) {
        uvs = uvs + [(positions[i * 3 + ua] - lo[ua]) / du,
                     (positions[i * 3 + va] - lo[va]) / dv];
    }
    return [positions, m[1], uvs, m[3]];
}

# Cylindrical around Y: u = angle/2pi, v = height. Seam triangles fixed by
# duplicating their wrapped corners with u+1.
fn cylindrical_project(m) {
    int b = 0; int dy = 0; int faces = 0; int hi = 0; int i = 0; int lo = 0; int normals = 0; int nv = 0; int positions = 0; int u = 0; int uvs = 0; int x = 0; int y = 0; int z = 0;
    b = mesh.bounds(m);
    lo = b[0];
    hi = b[1];
    positions = m[0];
    normals = m[1];
    faces = m[3];
    dy = mathx.maxv(hi[1] - lo[1], 1.0e-9);
    nv = len(positions) // 3;
    uvs = [];
    for (i = 0; i < nv; ++i) {
        x = positions[i * 3];
        y = positions[i * 3 + 1];
        z = positions[i * 3 + 2];
        u = mathx.atan2(z, x) / mathx.TAU + 0.5;
        uvs = uvs + [u, (y - lo[1]) / dy];
    }
    return fix_wrap_seam([positions, normals, uvs, faces]);
}

# Spherical about the origin: u = longitude, v = latitude.
fn spherical_project(m) {
    int faces = 0; int i = 0; int normals = 0; int nv = 0; int p = 0; int positions = 0; int u = 0; int uvs = 0; int v = 0;
    positions = m[0];
    normals = m[1];
    faces = m[3];
    nv = len(positions) // 3;
    uvs = [];
    for (i = 0; i < nv; ++i) {
        p = vector.vnormalize([positions[i * 3], positions[i * 3 + 1],
                               positions[i * 3 + 2]]);
        u = mathx.atan2(p[2], p[0]) / mathx.TAU + 0.5;
        v = acos(mathx.clampf(p[1], -1.0, 1.0)) / pi;
        uvs = uvs + [u, v];
    }
    return fix_wrap_seam([positions, normals, uvs, faces]);
}

# Duplicate vertices of triangles that straddle the u wrap seam, shifting the
# low-u corners by +1 so each triangle is locally continuous.
fn fix_wrap_seam(m) {
    int f = 0; int faces = 0; int hasN = 0; int i0 = 0; int i1 = 0; int i2 = 0; int idx = 0; int k = 0; int newIdx = 0; int nf = 0; int normals = 0; int positions = 0; int u = 0; int u0 = 0; int u1 = 0; int u2 = 0; int umax = 0; int umin = 0; int uvs = 0;
    positions = m[0];
    normals = m[1];
    uvs = m[2];
    faces = m[3];
    hasN = len(normals) > 0;
    nf = len(faces) // 3;
    for (f = 0; f < nf; ++f) {
        i0 = faces[f * 3];
        i1 = faces[f * 3 + 1];
        i2 = faces[f * 3 + 2];
        u0 = uvs[i0 * 2];
        u1 = uvs[i1 * 2];
        u2 = uvs[i2 * 2];
        umax = mathx.maxv(u0, mathx.maxv(u1, u2));
        umin = mathx.minv(u0, mathx.minv(u1, u2));
        if (umax - umin > 0.5) {
            # wrapping triangle: lift every low corner by +1 as a new vertex
            for (k = 0; k < 3; ++k) {
                idx = faces[f * 3 + k];
                u = uvs[idx * 2];
                if (u < 0.5) {
                    newIdx = len(positions) // 3;
                    positions = positions + [positions[idx * 3],
                                             positions[idx * 3 + 1],
                                             positions[idx * 3 + 2]];
                    if (hasN) {
                        normals = normals + [normals[idx * 3],
                                             normals[idx * 3 + 1],
                                             normals[idx * 3 + 2]];
                    }
                    uvs = uvs + [u + 1.0, uvs[idx * 2 + 1]];
                    faces[f * 3 + k] = newIdx;
                }
            }
        }
    }
    return [positions, normals, uvs, faces];
}

# Box (tri-planar) projection: each face is mapped along its dominant normal
# axis. Corners are unwelded first so faces are independent.
fn box_project(m) {
    int a = 0; int anx = 0; int any2 = 0; int anz = 0; int b = 0; int bb = 0; int c = 0; int f = 0; int faces = 0; int hi = 0; int ia = 0; int ib = 0; int ic = 0; int idx = 0; int idxs = 0; int k = 0; int lo = 0; int n = 0; int nf = 0; int normals = 0; int nv = 0; int positions = 0; int span = 0; int ua = 0; int um = 0; int uvs = 0; int va = 0;
    um = unweld(m);
    positions = um[0];
    normals = um[1];
    faces = um[3];
    b = mesh.bounds(um);
    lo = b[0];
    hi = b[1];
    span = [mathx.maxv(hi[0] - lo[0], 1.0e-9),
            mathx.maxv(hi[1] - lo[1], 1.0e-9),
            mathx.maxv(hi[2] - lo[2], 1.0e-9)];
    nv = len(positions) // 3;
    uvs = arrayx.zerosf(nv * 2);
    nf = len(faces) // 3;
    for (f = 0; f < nf; ++f) {
        ia = faces[f * 3];
        ib = faces[f * 3 + 1];
        ic = faces[f * 3 + 2];
        a = [positions[ia * 3], positions[ia * 3 + 1], positions[ia * 3 + 2]];
        bb = [positions[ib * 3], positions[ib * 3 + 1], positions[ib * 3 + 2]];
        c = [positions[ic * 3], positions[ic * 3 + 1], positions[ic * 3 + 2]];
        n = vector.vcross(vector.vsub(bb, a), vector.vsub(c, a));
        anx = abs(n[0]);
        any2 = abs(n[1]);
        anz = abs(n[2]);
        ua = 0; va = 1;                        # default: project along z
        if (anx >= any2 && anx >= anz) { ua = 2; va = 1; }
        elif (any2 >= anx && any2 >= anz) { ua = 0; va = 2; }
        idxs = [ia, ib, ic];
        for (k = 0; k < 3; ++k) {
            idx = idxs[k];
            uvs[idx * 2] = (positions[idx * 3 + ua] - lo[ua]) / span[ua];
            uvs[idx * 2 + 1] = (positions[idx * 3 + va] - lo[va]) / span[va];
        }
    }
    return [positions, normals, uvs, faces];
}

# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------

# Give every triangle corner its own vertex (3 * faceCount vertices).
fn unweld(m) {
    int count = 0; int faces = 0; int hasN = 0; int hasT = 0; int idx = 0; int k = 0; int nf2 = 0; int nn = 0; int normals = 0; int np = 0; int nt = 0; int positions = 0; int uvs = 0;
    positions = m[0];
    normals = m[1];
    uvs = m[2];
    faces = m[3];
    hasN = len(normals) > 0;
    hasT = len(uvs) > 0;
    np = [];
    nn = [];
    nt = [];
    nf2 = [];
    count = len(faces);
    for (k = 0; k < count; ++k) {
        idx = faces[k];
        np = np + [positions[idx * 3], positions[idx * 3 + 1], positions[idx * 3 + 2]];
        if (hasN) {
            nn = nn + [normals[idx * 3], normals[idx * 3 + 1], normals[idx * 3 + 2]];
        }
        if (hasT) {
            nt = nt + [uvs[idx * 2], uvs[idx * 2 + 1]];
        }
        nf2[k] = k;
    }
    return [np, nn, nt, nf2];
}

# ---------------------------------------------------------------------------
# Lightmap-style atlas: each face gets its own cell in a sqrt(nf) grid.
# Guaranteed overlap-free (at the cost of seams everywhere) - the classic
# baseline used for baking. margin is the fraction of each cell kept empty.
# ---------------------------------------------------------------------------

fn atlas_pack_faces(m, margin) {
    int cellSize = 0; int cells = 0; int cx = 0; int cy = 0; int f = 0; int faces = 0; int idx = 0; int inner = 0; int k = 0; int nf = 0; int normals = 0; int ox = 0; int oy = 0; int positions = 0; int um = 0; int us = 0; int uvs = 0; int vs = 0;
    if (margin == nil) { margin = 0.1; }
    um = unweld(m);
    positions = um[0];
    normals = um[1];
    faces = um[3];
    nf = len(faces) // 3;
    cells = int(top(sqrt(nf * 1.0)));
    if (cells < 1) { cells = 1; }
    cellSize = 1.0 / cells;
    inner = cellSize * (1.0 - 2.0 * margin);
    uvs = arrayx.zerosf((len(positions) // 3) * 2);
    for (f = 0; f < nf; ++f) {
        cx = f % cells;
        cy = f // cells;
        ox = cx * cellSize + cellSize * margin;
        oy = cy * cellSize + cellSize * margin;
        # right triangle in the cell; corner k of face f is vertex 3f+k
        us = [0.0, 1.0, 0.0];
        vs = [0.0, 0.0, 1.0];
        for (k = 0; k < 3; ++k) {
            idx = f * 3 + k;
            uvs[idx * 2] = ox + us[k] * inner;
            uvs[idx * 2 + 1] = oy + vs[k] * inner;
        }
    }
    return [positions, normals, uvs, faces];
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

fn uv_bounds(m) {
    int c = 0; int hi = 0; int i = 0; int lo = 0; int n = 0; int uvs = 0; int v = 0;
    uvs = m[2];
    n = len(uvs) // 2;
    lo = [1.0e30, 1.0e30];
    hi = [-1.0e30, -1.0e30];
    for (i = 0; i < n; ++i) {
        for (c = 0; c < 2; ++c) {
            v = uvs[i * 2 + c];
            if (v < lo[c]) { lo[c] = v; }
            if (v > hi[c]) { hi[c] = v; }
        }
    }
    return [lo, hi];
}

fn scale_uvs(m, su, sv) {
    int i = 0; int n = 0; int out = 0; int uvs = 0;
    uvs = m[2];
    out = [];
    n = len(uvs) // 2;
    for (i = 0; i < n; ++i) {
        out[i * 2] = uvs[i * 2] * su;
        out[i * 2 + 1] = uvs[i * 2 + 1] * sv;
    }
    return [m[0], m[1], out, m[3]];
}

# Fraction of UV area outside [0,1]^2 corners plus per-face UV area sum -
# quick sanity number for an unwrap (areaSum <= 1 means it could fit a
# texture without overlap IF the layout is overlap-free).
fn uv_area_used(m) {
    int ax = 0; int ay = 0; int bx = 0; int by = 0; int f = 0; int faces = 0; int i0 = 0; int i1 = 0; int i2 = 0; int nf = 0; int total = 0; int uvs = 0;
    uvs = m[2];
    faces = m[3];
    nf = len(faces) // 3;
    total = 0.0;
    for (f = 0; f < nf; ++f) {
        i0 = faces[f * 3];
        i1 = faces[f * 3 + 1];
        i2 = faces[f * 3 + 2];
        ax = uvs[i1 * 2] - uvs[i0 * 2];
        ay = uvs[i1 * 2 + 1] - uvs[i0 * 2 + 1];
        bx = uvs[i2 * 2] - uvs[i0 * 2];
        by = uvs[i2 * 2 + 1] - uvs[i0 * 2 + 1];
        total = total + abs(ax * by - ay * bx) / 2.0;
    }
    return total;
}

# Generate a labelled checker texture useful for inspecting unwraps.
fn checker_test_pattern(size, cells) {
    int img = 0;
    img = texture.checker(size, size, cells, [0.95, 0.95, 0.95], [0.2, 0.2, 0.2]);
    # add a red U axis along the bottom and a green V axis along the left
    img = image.fill_rect(img, 0, 0, size, 2, [0.9, 0.1, 0.1]);
    img = image.fill_rect(img, 0, 0, 2, size, [0.1, 0.8, 0.1]);
    return img;
}
