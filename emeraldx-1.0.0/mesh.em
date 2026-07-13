# =============================================================================
# mesh.em - triangle meshes
#
# mesh = [positions, normals, uvs, faces]
#   positions  flat [x,y,z, ...]          len = 3 * vertexCount
#   normals    flat [x,y,z, ...]          len = 3 * vertexCount (may be [])
#   uvs        flat [u,v, ...]            len = 2 * vertexCount (may be [])
#   faces      flat [i0,i1,i2, ...]       triangle vertex indices, CCW
#
# One shared index space (positions/normals/uvs run parallel). OBJ files with
# per-corner v/vt/vn triples are unified on import.
# =============================================================================

import mathx;
import arrayx;
import strx;
import vector;
import mat4;
import dictx;
# emx-scope-safe

fn new_mesh() { return [[], [], [], []]; }

fn vertex_count(m) { return len(m[0]) // 3; }
fn face_count(m) { return len(m[3]) // 3; }

fn get_position(m, i) {
    int p = 0;
    p = m[0];
    return [p[i * 3], p[i * 3 + 1], p[i * 3 + 2]];
}

fn get_normal(m, i) {
    int nrm = 0;
    nrm = m[1];
    if (len(nrm) == 0) { return [0.0, 0.0, 1.0]; }
    return [nrm[i * 3], nrm[i * 3 + 1], nrm[i * 3 + 2]];
}

fn get_uv(m, i) {
    int t = 0;
    t = m[2];
    if (len(t) == 0) { return [0.0, 0.0]; }
    return [t[i * 2], t[i * 2 + 1]];
}

fn get_face(m, f) {
    int fc = 0;
    fc = m[3];
    return [fc[f * 3], fc[f * 3 + 1], fc[f * 3 + 2]];
}

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

# Axis-aligned cube (size s), 24 vertices so each face has flat normals and
# a full 0..1 UV square.
fn cube(s) {
    int base = 0; int corner = 0; int cu = 0; int cv = 0; int d = 0; int dirs = 0; int f = 0; int faces = 0; int hs = 0; int normals = 0; int nrm = 0; int positions = 0; int px = 0; int py = 0; int pz = 0; int ua = 0; int uvs = 0; int va = 0;
    hs = s / 2.0;
    positions = [];
    normals = [];
    uvs = [];
    faces = [];
    # per face: normal, u axis, v axis
    dirs = [
        [[0.0, 0.0, 1.0],  [1.0, 0.0, 0.0],  [0.0, 1.0, 0.0]],   # +z
        [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],   # -z
        [[1.0, 0.0, 0.0],  [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],   # +x
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0],  [0.0, 1.0, 0.0]],   # -x
        [[0.0, 1.0, 0.0],  [1.0, 0.0, 0.0],  [0.0, 0.0, -1.0]],  # +y
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0],  [0.0, 0.0, 1.0]]    # -y
    ];
    for (f = 0; f < 6; ++f) {
        d = dirs[f];
        nrm = d[0];
        ua = d[1];
        va = d[2];
        base = f * 4;
        for (corner = 0; corner < 4; ++corner) {
            cu = -1.0;
            cv = -1.0;
            if (corner == 1 || corner == 2) { cu = 1.0; }
            if (corner == 2 || corner == 3) { cv = 1.0; }
            px = nrm[0] * hs + ua[0] * hs * cu + va[0] * hs * cv;
            py = nrm[1] * hs + ua[1] * hs * cu + va[1] * hs * cv;
            pz = nrm[2] * hs + ua[2] * hs * cu + va[2] * hs * cv;
            positions = positions + [px, py, pz];
            normals = normals + [nrm[0], nrm[1], nrm[2]];
            uvs = uvs + [(cu + 1.0) / 2.0, (cv + 1.0) / 2.0];
        }
        faces = faces + [base, base + 1, base + 2, base, base + 2, base + 3];
    }
    return [positions, normals, uvs, faces];
}

# XZ plane grid centered at origin: size sx x sz, (nx x nz) cells, +Y normal.
fn plane(sx, sz, nx, nz) {
    int a = 0; int b = 0; int c = 0; int d = 0; int faces = 0; int i = 0; int j = 0; int normals = 0; int positions = 0; int stride = 0; int u = 0; int uvs = 0; int v = 0;
    positions = [];
    normals = [];
    uvs = [];
    faces = [];
    for (j = 0; j <= nz; ++j) {
        for (i = 0; i <= nx; ++i) {
            u = i * 1.0 / nx;
            v = j * 1.0 / nz;
            positions = positions + [(u - 0.5) * sx, 0.0, (v - 0.5) * sz];
            normals = normals + [0.0, 1.0, 0.0];
            uvs = uvs + [u, v];
        }
    }
    stride = nx + 1;
    for (j = 0; j < nz; ++j) {
        for (i = 0; i < nx; ++i) {
            a = j * stride + i;
            b = a + 1;
            c = a + stride;
            d = c + 1;
            faces = faces + [a, c, b, b, c, d];
        }
    }
    return [positions, normals, uvs, faces];
}

# Latitude/longitude sphere with proper spherical UVs.
fn uv_sphere(radius, segments, rings) {
    int a = 0; int b = 0; int c = 0; int cp = 0; int d = 0; int faces = 0; int normals = 0; int nx = 0; int ny = 0; int nz = 0; int phi = 0; int positions = 0; int r = 0; int sgm = 0; int sp = 0; int stride = 0; int theta = 0; int uvs = 0;
    positions = [];
    normals = [];
    uvs = [];
    faces = [];
    for (r = 0; r <= rings; ++r) {
        phi = pi * r / rings;                  # 0..pi from +Y pole
        sp = sin(phi);
        cp = cos(phi);
        for (sgm = 0; sgm <= segments; ++sgm) {
            theta = mathx.TAU * sgm / segments;
            nx = sp * cos(theta);
            ny = cp;
            nz = sp * sin(theta);
            positions = positions + [nx * radius, ny * radius, nz * radius];
            normals = normals + [nx, ny, nz];
            uvs = uvs + [sgm * 1.0 / segments, r * 1.0 / rings];
        }
    }
    stride = segments + 1;
    for (r = 0; r < rings; ++r) {
        for (sgm = 0; sgm < segments; ++sgm) {
            a = r * stride + sgm;
            b = a + 1;
            c = a + stride;
            d = c + 1;
            if (r > 0) { faces = faces + [a, b, c]; }
            if (r < rings - 1) { faces = faces + [b, d, c]; }
        }
    }
    return [positions, normals, uvs, faces];
}

# Icosphere: subdivided icosahedron - near-uniform triangles.
fn icosphere(radius, subdivisions) {
    int a = 0; int ab = 0; int b = 0; int base = 0; int bc = 0; int c = 0; int ca = 0; int cache = 0; int f = 0; int faces = 0; int i = 0; int newFaces = 0; int nf = 0; int normals = 0; int nv = 0; int p = 0; int positions = 0; int px = 0; int py = 0; int pz = 0; int r1 = 0; int r2 = 0; int r3 = 0; int s = 0; int t = 0; int uvs = 0;
    t = (1.0 + sqrt(5.0)) / 2.0;
    base = [
        [-1.0, t, 0.0], [1.0, t, 0.0], [-1.0, 0.0 - t, 0.0], [1.0, 0.0 - t, 0.0],
        [0.0, -1.0, t], [0.0, 1.0, t], [0.0, -1.0, 0.0 - t], [0.0, 1.0, 0.0 - t],
        [t, 0.0, -1.0], [t, 0.0, 1.0], [0.0 - t, 0.0, -1.0], [0.0 - t, 0.0, 1.0]
    ];
    positions = [];
    for (i = 0; i < 12; ++i) {
        p = vector.vnormalize(base[i]);
        positions = positions + [p[0], p[1], p[2]];
    }
    faces = [0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11,
             1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8,
             3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9,
             4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1];
    for (s = 0; s < subdivisions; ++s) {
        cache = dictx.dnew();
        newFaces = [];
        nf = len(faces) // 3;
        for (f = 0; f < nf; ++f) {
            a = faces[f * 3];
            b = faces[f * 3 + 1];
            c = faces[f * 3 + 2];
            r1 = midpoint_index(positions, cache, a, b);
            positions = r1[0];
            cache = r1[1];
            ab = r1[2];
            r2 = midpoint_index(positions, cache, b, c);
            positions = r2[0];
            cache = r2[1];
            bc = r2[2];
            r3 = midpoint_index(positions, cache, c, a);
            positions = r3[0];
            cache = r3[1];
            ca = r3[2];
            newFaces = newFaces + [a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca];
        }
        faces = newFaces;
    }
    # normals = normalized positions; spherical uvs
    normals = [];
    uvs = [];
    nv = len(positions) // 3;
    for (i = 0; i < nv; ++i) {
        px = positions[i * 3];
        py = positions[i * 3 + 1];
        pz = positions[i * 3 + 2];
        normals = normals + [px, py, pz];
        uvs = uvs + [0.5 + mathx.atan2(pz, px) / mathx.TAU, acos(mathx.clampf(py, -1.0, 1.0)) / pi];
        positions[i * 3] = px * radius;
        positions[i * 3 + 1] = py * radius;
        positions[i * 3 + 2] = pz * radius;
    }
    return [positions, normals, uvs, faces];
}

# midpoint vertex index with edge cache. Returns [positions, cache, idx].
fn midpoint_index(positions, cache, a, b) {
    int hi = 0; int idx = 0; int key = 0; int lo = 0; int mx = 0; int my = 0; int mz = 0; int p = 0;
    lo = mathx.mini(a, b);
    hi = mathx.maxi(a, b);
    key = "{lo}_{hi}";
    if (dictx.dhas(cache, key)) {
        return [positions, cache, dictx.dget(cache, key, 0)];
    }
    mx = (positions[lo * 3] + positions[hi * 3]) / 2.0;
    my = (positions[lo * 3 + 1] + positions[hi * 3 + 1]) / 2.0;
    mz = (positions[lo * 3 + 2] + positions[hi * 3 + 2]) / 2.0;
    p = vector.vnormalize([mx, my, mz]);
    idx = len(positions) // 3;
    positions = positions + [p[0], p[1], p[2]];
    cache = dictx.dset(cache, key, idx);
    return [positions, cache, idx];
}

# Torus around Y axis: major radius R, tube radius r.
fn torus(bigR, tubeR, segments, sides) {
    int a = 0; int b = 0; int cu = 0; int cv = 0; int faces = 0; int i = 0; int j = 0; int normals = 0; int positions = 0; int stride = 0; int su = 0; int sv = 0; int u = 0; int uvs = 0; int v = 0;
    positions = [];
    normals = [];
    uvs = [];
    faces = [];
    for (i = 0; i <= segments; ++i) {
        u = mathx.TAU * i / segments;
        cu = cos(u);
        su = sin(u);
        for (j = 0; j <= sides; ++j) {
            v = mathx.TAU * j / sides;
            cv = cos(v);
            sv = sin(v);
            positions = positions + [(bigR + tubeR * cv) * cu,
                                     tubeR * sv,
                                     (bigR + tubeR * cv) * su];
            normals = normals + [cv * cu, sv, cv * su];
            uvs = uvs + [i * 1.0 / segments, j * 1.0 / sides];
        }
    }
    stride = sides + 1;
    for (i = 0; i < segments; ++i) {
        for (j = 0; j < sides; ++j) {
            a = i * stride + j;
            b = a + stride;
            faces = faces + [a, b, a + 1, a + 1, b, b + 1];
        }
    }
    return [positions, normals, uvs, faces];
}

# Open cylinder along Y (no caps), radius r, height h.
fn cylinder(r, h, segments) {
    int a = 0; int b = 0; int c = 0; int cx = 0; int cz = 0; int d = 0; int faces = 0; int i = 0; int j = 0; int normals = 0; int positions = 0; int theta = 0; int uvs = 0; int y = 0;
    positions = [];
    normals = [];
    uvs = [];
    faces = [];
    for (i = 0; i <= segments; ++i) {
        theta = mathx.TAU * i / segments;
        cx = cos(theta);
        cz = sin(theta);
        for (j = 0; j <= 1; ++j) {
            y = h * (j - 0.5);
            positions = positions + [r * cx, y, r * cz];
            normals = normals + [cx, 0.0, cz];
            uvs = uvs + [i * 1.0 / segments, j * 1.0];
        }
    }
    for (i = 0; i < segments; ++i) {
        a = i * 2;
        b = a + 1;
        c = a + 2;
        d = a + 3;
        faces = faces + [a, c, b, b, c, d];
    }
    return [positions, normals, uvs, faces];
}

# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

# Area-weighted smooth vertex normals from face geometry.
fn compute_normals(m) {
    int acc = 0; int ax = 0; int ay = 0; int az = 0; int bx = 0; int by = 0; int bz = 0; int cx = 0; int cy = 0; int cz = 0; int f = 0; int faces = 0; int i = 0; int ia = 0; int ib = 0; int ic = 0; int l = 0; int nf = 0; int nv = 0; int nx = 0; int ny = 0; int nz = 0; int positions = 0; int ux = 0; int uy = 0; int uz = 0; int vx = 0; int vy = 0; int vz = 0;
    positions = m[0];
    faces = m[3];
    nv = len(positions) // 3;
    acc = arrayx.zerosf(nv * 3);
    nf = len(faces) // 3;
    for (f = 0; f < nf; ++f) {
        ia = faces[f * 3];
        ib = faces[f * 3 + 1];
        ic = faces[f * 3 + 2];
        ax = positions[ia * 3];     ay = positions[ia * 3 + 1]; az = positions[ia * 3 + 2];
        bx = positions[ib * 3];     by = positions[ib * 3 + 1]; bz = positions[ib * 3 + 2];
        cx = positions[ic * 3];     cy = positions[ic * 3 + 1]; cz = positions[ic * 3 + 2];
        ux = bx - ax; uy = by - ay; uz = bz - az;
        vx = cx - ax; vy = cy - ay; vz = cz - az;
        nx = uy * vz - uz * vy;      # cross product = 2*area weighted normal
        ny = uz * vx - ux * vz;
        nz = ux * vy - uy * vx;
        acc[ia * 3] = acc[ia * 3] + nx;
        acc[ia * 3 + 1] = acc[ia * 3 + 1] + ny;
        acc[ia * 3 + 2] = acc[ia * 3 + 2] + nz;
        acc[ib * 3] = acc[ib * 3] + nx;
        acc[ib * 3 + 1] = acc[ib * 3 + 1] + ny;
        acc[ib * 3 + 2] = acc[ib * 3 + 2] + nz;
        acc[ic * 3] = acc[ic * 3] + nx;
        acc[ic * 3 + 1] = acc[ic * 3 + 1] + ny;
        acc[ic * 3 + 2] = acc[ic * 3 + 2] + nz;
    }
    for (i = 0; i < nv; ++i) {
        nx = acc[i * 3];
        ny = acc[i * 3 + 1];
        nz = acc[i * 3 + 2];
        l = sqrt(nx * nx + ny * ny + nz * nz);
        if (l < 1.0e-12) { l = 1.0; nz = 1.0; }
        acc[i * 3] = nx / l;
        acc[i * 3 + 1] = ny / l;
        acc[i * 3 + 2] = nz / l;
    }
    return [positions, acc, m[2], faces];
}

# Apply a mat4 to positions (points) and its normal matrix to normals.
fn transform_mesh(m, mat) {
    int i = 0; int nm = 0; int nn = 0; int normals = 0; int np = 0; int nv = 0; int p = 0; int positions = 0; int v = 0; int w = 0;
    positions = m[0];
    normals = m[1];
    nv = len(positions) // 3;
    nm = mat4.normal_matrix(mat);
    np = [];
    nn = [];
    for (i = 0; i < nv; ++i) {
        p = mat4.transform_point(mat, [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]]);
        np = np + [p[0], p[1], p[2]];
        if (len(normals) > 0) {
            v = [normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]];
            w = vector.vnormalize(mat4.transform_dir(nm, v));
            nn = nn + [w[0], w[1], w[2]];
        }
    }
    return [np, nn, m[2], m[3]];
}

# Combine two meshes into one.
fn merge(a, b) {
    int bf = 0; int faces = 0; int i = 0; int normals = 0; int offset = 0; int positions = 0; int uvs = 0;
    offset = len(a[0]) // 3;
    positions = a[0] + b[0];
    normals = a[1] + b[1];
    uvs = a[2] + b[2];
    faces = a[3];
    bf = b[3];
    for (i = 0; i < len(bf); ++i) { faces[len(faces)] = bf[i] + offset; }
    return [positions, normals, uvs, faces];
}

# Axis-aligned bounds. Returns [min3, max3].
fn bounds(m) {
    int c = 0; int hi = 0; int i = 0; int lo = 0; int nv = 0; int positions = 0; int v = 0;
    positions = m[0];
    nv = len(positions) // 3;
    lo = [1.0e30, 1.0e30, 1.0e30];
    hi = [-1.0e30, -1.0e30, -1.0e30];
    for (i = 0; i < nv; ++i) {
        for (c = 0; c < 3; ++c) {
            v = positions[i * 3 + c];
            if (v < lo[c]) { lo[c] = v; }
            if (v > hi[c]) { hi[c] = v; }
        }
    }
    return [lo, hi];
}

fn surface_area(m) {
    int a = 0; int b = 0; int c = 0; int cr = 0; int f = 0; int faces = 0; int nf = 0; int positions = 0; int total = 0;
    positions = m[0];
    faces = m[3];
    nf = len(faces) // 3;
    total = 0.0;
    for (f = 0; f < nf; ++f) {
        a = get_position(m, faces[f * 3]);
        b = get_position(m, faces[f * 3 + 1]);
        c = get_position(m, faces[f * 3 + 2]);
        cr = vector.vcross(vector.vsub(b, a), vector.vsub(c, a));
        total = total + vector.vnorm(cr) / 2.0;
    }
    return total;
}

# ---------------------------------------------------------------------------
# Wavefront OBJ I/O
# ---------------------------------------------------------------------------

fn save_obj(m, path) {
    int f = 0; int faces = 0; int hasN = 0; int hasT = 0; int i = 0; int idx = 0; int k = 0; int line = 0; int nf = 0; int normals = 0; int nv = 0; int positions = 0; int text = 0; int uvs = 0; int x = 0; int y = 0; int z = 0;
    positions = m[0];
    normals = m[1];
    uvs = m[2];
    faces = m[3];
    nv = len(positions) // 3;
    hasN = len(normals) > 0;
    hasT = len(uvs) > 0;
    text = "# emeraldx mesh\n";
    for (i = 0; i < nv; ++i) {
        x = positions[i * 3];
        y = positions[i * 3 + 1];
        z = positions[i * 3 + 2];
        text = text + "v " + strx.fmt_g(x, 6) + " " + strx.fmt_g(y, 6) + " " + strx.fmt_g(z, 6) + "\n";
    }
    if (hasT) {
        for (i = 0; i < nv; ++i) {
            text = text + "vt " + strx.fmt_g(uvs[i * 2], 6) + " " + strx.fmt_g(uvs[i * 2 + 1], 6) + "\n";
        }
    }
    if (hasN) {
        for (i = 0; i < nv; ++i) {
            text = text + "vn " + strx.fmt_g(normals[i * 3], 6) + " " + strx.fmt_g(normals[i * 3 + 1], 6) + " " + strx.fmt_g(normals[i * 3 + 2], 6) + "\n";
        }
    }
    nf = len(faces) // 3;
    for (f = 0; f < nf; ++f) {
        line = "f";
        for (k = 0; k < 3; ++k) {
            idx = faces[f * 3 + k] + 1;
            if (hasT && hasN) { line = line + " {idx}/{idx}/{idx}"; }
            elif (hasT) { line = line + " {idx}/{idx}"; }
            elif (hasN) { line = line + " {idx}//{idx}"; }
            else { line = line + " {idx}"; }
        }
        text = text + line + "\n";
    }
    File fh = path;
    fh.write(text);
    return true;
}

# Load an OBJ (v / vt / vn / f, triangles or fans). Unifies v/vt/vn corner
# triples into a single index space.
fn load_obj(path) {
    int corner = 0; int faces = 0; int k = 0; int key = 0; int kind = 0; int lines = 0; int newIdx = 0; int ni = 0; int nlist = 0; int normals = 0; int nv2 = 0; int parts = 0; int positions = 0; int pv = 0; int seen = 0; int t = 0; int text = 0; int ti = 0; int tlist = 0; int trip = 0; int tv = 0; int uvs = 0; int vi = 0; int vlist = 0;
    File fh = path;
    text = fh.read();
    if (text == nil || len(text) == 0) { return nil; }
    vlist = [];
    tlist = [];
    nlist = [];
    positions = [];
    normals = [];
    uvs = [];
    faces = [];
    seen = dictx.dnew();
    lines = strx.split_lines(text);
    for ln in lines {
        t = strx.trim(ln);
        if (len(t) == 0 || t[0] == '#') { continue; }
        parts = strx.split_ws(t);
        kind = parts[0];
        if (kind == "v") {
            vlist[len(vlist)] = [strx.parse_float(parts[1]), strx.parse_float(parts[2]), strx.parse_float(parts[3])];
        } elif (kind == "vt") {
            tlist[len(tlist)] = [strx.parse_float(parts[1]), strx.parse_float(parts[2])];
        } elif (kind == "vn") {
            nlist[len(nlist)] = [strx.parse_float(parts[1]), strx.parse_float(parts[2]), strx.parse_float(parts[3])];
        } elif (kind == "f") {
            corner = [];
            for (k = 1; k < len(parts); ++k) {
                key = parts[k];
                if (dictx.dhas(seen, key)) {
                    corner[len(corner)] = dictx.dget(seen, key, 0);
                } else {
                    trip = parse_obj_corner(key);
                    vi = trip[0];
                    ti = trip[1];
                    ni = trip[2];
                    newIdx = len(positions) // 3;
                    pv = vlist[vi - 1];
                    positions = positions + [pv[0], pv[1], pv[2]];
                    if (ti > 0 && len(tlist) >= ti) {
                        tv = tlist[ti - 1];
                        uvs = uvs + [tv[0], tv[1]];
                    } elif (len(tlist) > 0) {
                        uvs = uvs + [0.0, 0.0];
                    }
                    if (ni > 0 && len(nlist) >= ni) {
                        nv2 = nlist[ni - 1];
                        normals = normals + [nv2[0], nv2[1], nv2[2]];
                    } elif (len(nlist) > 0) {
                        normals = normals + [0.0, 0.0, 1.0];
                    }
                    seen = dictx.dset(seen, key, newIdx);
                    corner[len(corner)] = newIdx;
                }
            }
            # triangulate as a fan
            for (k = 2; k < len(corner); ++k) {
                faces = faces + [corner[0], corner[k - 1], corner[k]];
            }
        }
    }
    return [positions, normals, uvs, faces];
}

# "7/3/5" | "7//5" | "7/3" | "7" -> [v, vt, vn] (0 = missing).
fn parse_obj_corner(s) {
    int a = 0; int nrm = 0; int t = 0; int v = 0;
    a = strx.split(s, "/");
    v = strx.parse_int(a[0]);
    t = 0;
    nrm = 0;
    if (len(a) > 1 && len(a[1]) > 0) { t = strx.parse_int(a[1]); }
    if (len(a) > 2 && len(a[2]) > 0) { nrm = strx.parse_int(a[2]); }
    return [v, t, nrm];
}
