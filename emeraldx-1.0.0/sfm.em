# =============================================================================
# sfm.em - structure from motion / photogrammetry
#
#   K = sfm.intrinsics(f, cx, cy)
#   x = sfm.project(K, R, t, X)                     pinhole projection
#   F/E: eight_point(ptsA, ptsB [, Ka, Kb])          normalized 8-point
#   pose = recover_pose(E, ptsA, ptsB, K)            R,t with cheirality test
#   X = triangulate(Ka, Ra, ta, Kb, Rb, tb, xa, xb)  DLT (midpoint-quality)
#   disparity_map(left, right, ...)                  stereo block matching
#   save_ply_points(points, colors, path)            point cloud export
#
# Conventions: world point X, camera x = K [R | t] X; R,t map world -> cam.
# Pixel points are [x, y]. All estimation is least squares via the smallest
# singular vector (matrix.null_vector).
# =============================================================================

import mathx;
import arrayx;
import vector;
import matrix;
import strx;
import image;
# emx-scope-safe

fn intrinsics(f, cx, cy) {
    return [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]];
}

# Project world point X with K, R (3x3), t (3). Returns [u, v] or nil if
# behind the camera.
fn project(k, r, t, x) {
    int pc = 0; int uvw = 0;
    pc = vector.vadd(matrix.matvec(r, x), t);
    if (pc[2] <= 1.0e-9) { return nil; }
    uvw = matrix.matvec(k, pc);
    return [uvw[0] / uvw[2], uvw[1] / uvw[2]];
}

# Pixel -> normalized camera coords (z = 1 plane).
fn normalize_point(k, p) {
    int cx = 0; int cy = 0; int fx = 0; int fy = 0;
    fx = k[0][0];
    fy = k[1][1];
    cx = k[0][2];
    cy = k[1][2];
    return [(p[0] - cx) / fx, (p[1] - cy) / fy, 1.0];
}

# ---------------------------------------------------------------------------
# Hartley normalization: translate centroid to origin, scale mean distance
# to sqrt(2). Returns [normalizedPoints, T3x3].
# ---------------------------------------------------------------------------

fn hartley_normalize(pts) {
    int dx = 0; int dy = 0; int i = 0; int md = 0; int mx = 0; int my = 0; int n = 0; int out = 0; int s = 0; int t = 0;
    n = len(pts);
    mx = 0.0;
    my = 0.0;
    for (i = 0; i < n; ++i) {
        mx = mx + pts[i][0];
        my = my + pts[i][1];
    }
    mx = mx / n;
    my = my / n;
    md = 0.0;
    for (i = 0; i < n; ++i) {
        dx = pts[i][0] - mx;
        dy = pts[i][1] - my;
        md = md + sqrt(dx * dx + dy * dy);
    }
    md = md / n;
    s = sqrt(2.0) / mathx.maxv(md, 1.0e-12);
    t = [[s, 0.0, 0.0 - s * mx], [0.0, s, 0.0 - s * my], [0.0, 0.0, 1.0]];
    out = [];
    for (i = 0; i < n; ++i) {
        out[i] = [s * (pts[i][0] - mx), s * (pts[i][1] - my)];
    }
    return [out, t];
}

# ---------------------------------------------------------------------------
# Normalized 8-point algorithm. With Ka/Kb given, points are pixel coords and
# the ESSENTIAL matrix is returned; with nil intrinsics the FUNDAMENTAL
# matrix of the raw points is returned. Needs >= 8 correspondences.
# ---------------------------------------------------------------------------

fn eight_point(ptsA, ptsB, ka, kb) {
    int a2 = 0; int avg = 0; int b2 = 0; int f = 0; int fv = 0; int i = 0; int m = 0; int n = 0; int na = 0; int nb = 0; int pa = 0; int pb = 0; int ra = 0; int rb = 0; int s = 0; int s2 = 0; int sv = 0; int ta = 0; int tb = 0; int u = 0; int vt = 0; int x = 0; int xp = 0; int y = 0; int yp = 0;
    n = len(ptsA);
    a2 = ptsA;
    b2 = ptsB;
    if (ka != nil) {
        a2 = [];
        b2 = [];
        for (i = 0; i < n; ++i) {
            na = normalize_point(ka, ptsA[i]);
            nb = normalize_point(kb, ptsB[i]);
            a2[i] = [na[0], na[1]];
            b2[i] = [nb[0], nb[1]];
        }
    }
    ra = hartley_normalize(a2);
    rb = hartley_normalize(b2);
    pa = ra[0];
    pb = rb[0];
    ta = ra[1];
    tb = rb[1];
    # rows of the constraint matrix: x'^T F x = 0 with x from A, x' from B
    m = [];
    for (i = 0; i < n; ++i) {
        x = pa[i][0];
        y = pa[i][1];
        xp = pb[i][0];
        yp = pb[i][1];
        m[i] = [xp * x, xp * y, xp, yp * x, yp * y, yp, x, y, 1.0];
    }
    fv = matrix.null_vector(m);
    f = [[fv[0], fv[1], fv[2]], [fv[3], fv[4], fv[5]], [fv[6], fv[7], fv[8]]];
    # denormalize FIRST: F = Tb^T F Ta. The singular-value structure below
    # must be enforced in the ORIGINAL frame - it does not survive the
    # (anisotropic-in-effect) Hartley similarity transforms.
    f = matrix.matmul(matrix.transpose(tb), matrix.matmul(f, ta));
    sv = matrix.svd(f);
    u = sv[0];
    s = sv[1];
    vt = matrix.transpose(sv[2]);        # svd returns V; we need V^T
    s2 = [[s[0], 0.0, 0.0], [0.0, s[1], 0.0], [0.0, 0.0, 0.0]];
    if (ka != nil) {
        # essential matrices have two equal singular values
        avg = (s[0] + s[1]) * 0.5;
        s2 = [[avg, 0.0, 0.0], [0.0, avg, 0.0], [0.0, 0.0, 0.0]];
    }
    f = matrix.matmul(u, matrix.matmul(s2, vt));
    # fix overall scale (F/E are homogeneous)
    if (abs(f[2][2]) > 1.0e-6) { f = matrix.scale(f, 1.0 / f[2][2]); }
    return f;
}

# Epipolar residual |x'^T F x| averaged over correspondences (diagnostic).
fn epipolar_error(f, ptsA, ptsB) {
    int fx = 0; int i = 0; int n = 0; int s = 0; int xa = 0; int xb = 0;
    n = len(ptsA);
    s = 0.0;
    for (i = 0; i < n; ++i) {
        xa = [ptsA[i][0], ptsA[i][1], 1.0];
        xb = [ptsB[i][0], ptsB[i][1], 1.0];
        fx = matrix.matvec(f, xa);
        s = s + abs(vector.vdot(xb, fx));
    }
    return s / n;
}

# ---------------------------------------------------------------------------
# Pose from an essential matrix: E = [t]x R. Four candidates; the one that
# puts the (triangulated) points in front of BOTH cameras wins.
# ptsA/ptsB are pixel points; camera A is taken as identity.
# Returns [R, t, inFrontCount].
# ---------------------------------------------------------------------------

fn recover_pose(e, ptsA, ptsB, k) {
    int best = 0; int bestCount = 0; int c = 0; int cand = 0; int cands = 0; int chosen = 0; int count = 0; int i = 0; int ident = 0; int n = 0; int pb = 0; int r1 = 0; int r2 = 0; int sv = 0; int tcol = 0; int tneg = 0; int u = 0; int vt = 0; int w = 0; int x = 0; int zero3 = 0;
    sv = matrix.svd(e);
    u = sv[0];
    vt = matrix.transpose(sv[2]);        # svd returns V; we need V^T
    # ensure proper rotations
    if (matrix.det(u) < 0.0) { u = matrix.scale(u, -1.0); }
    if (matrix.det(vt) < 0.0) { vt = matrix.scale(vt, -1.0); }
    w = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    r1 = matrix.matmul(u, matrix.matmul(w, vt));
    r2 = matrix.matmul(u, matrix.matmul(matrix.transpose(w), vt));
    tcol = [u[0][2], u[1][2], u[2][2]];
    tneg = [0.0 - tcol[0], 0.0 - tcol[1], 0.0 - tcol[2]];
    cands = [[r1, tcol], [r1, tneg], [r2, tcol], [r2, tneg]];
    ident = matrix.identity(3);
    zero3 = [0.0, 0.0, 0.0];
    best = 0;
    bestCount = -1;
    for (c = 0; c < 4; ++c) {
        cand = cands[c];
        count = 0;
        n = len(ptsA);
        for (i = 0; i < n; ++i) {
            x = triangulate(k, ident, zero3, k, cand[0], cand[1], ptsA[i], ptsB[i]);
            if (x == nil) { continue; }
            # in front of camera A?
            if (x[2] > 0.0) {
                # and in front of camera B?
                pb = vector.vadd(matrix.matvec(cand[0], x), cand[1]);
                if (pb[2] > 0.0) { count = count + 1; }
            }
        }
        if (count > bestCount) { bestCount = count; best = c; }
    }
    chosen = cands[best];
    return [chosen[0], chosen[1], bestCount];
}

# ---------------------------------------------------------------------------
# DLT triangulation from two views. xa/xb are pixel points.
# ---------------------------------------------------------------------------

fn projection_matrix(k, r, t) {
    int i = 0; int j = 0; int p = 0; int row = 0;
    p = [];
    for (i = 0; i < 3; ++i) {
        row = [];
        for (j = 0; j < 3; ++j) { row[j] = r[i][j]; }
        row[3] = t[i];
        p[i] = row;
    }
    return matrix.matmul(k, p);       # 3x3 * 3x4 -> 3x4
}

fn triangulate(ka, ra, ta, kb, rb, tb, xa, xb) {
    int h = 0; int m = 0; int pa = 0; int pb = 0;
    pa = projection_matrix(ka, ra, ta);
    pb = projection_matrix(kb, rb, tb);
    m = [];
    m[0] = row_combo(pa, 2, xa[0], 0);
    m[1] = row_combo(pa, 2, xa[1], 1);
    m[2] = row_combo(pb, 2, xb[0], 0);
    m[3] = row_combo(pb, 2, xb[1], 1);
    h = matrix.null_vector(m);
    if (abs(h[3]) < 1.0e-12) { return nil; }
    return [h[0] / h[3], h[1] / h[3], h[2] / h[3]];
}

# helper: coef*P[rowScale] - P[rowPick]  (a DLT constraint row, length 4)
fn row_combo(p, rowScale, coef, rowPick) {
    int j = 0; int out = 0; int rp = 0; int rs = 0;
    rs = p[rowScale];
    rp = p[rowPick];
    out = [];
    for (j = 0; j < 4; ++j) { out[j] = coef * rs[j] - rp[j]; }
    return out;
}

# Reprojection RMS error of triangulated points (diagnostic).
fn reprojection_error(k, r, t, points, pixels) {
    int cnt = 0; int dx = 0; int dy = 0; int i = 0; int n = 0; int s = 0; int uv = 0;
    n = len(points);
    s = 0.0;
    cnt = 0;
    for (i = 0; i < n; ++i) {
        uv = project(k, r, t, points[i]);
        if (uv == nil) { continue; }
        dx = uv[0] - pixels[i][0];
        dy = uv[1] - pixels[i][1];
        s = s + dx * dx + dy * dy;
        cnt = cnt + 1;
    }
    if (cnt == 0) { return 0.0; }
    return sqrt(s / cnt);
}

# ---------------------------------------------------------------------------
# Stereo block matching for RECTIFIED pairs (gray images). SAD cost.
# Returns a gray image whose values are disparity / maxDisparity.
# ---------------------------------------------------------------------------

fn disparity_map(left, right, blockRadius, maxDisparity) {
    int bestCost = 0; int bestD = 0; int bx = 0; int by = 0; int cost = 0; int disp = 0; int dmax = 0; int dv = 0; int h = 0; int lbase = 0; int ld = 0; int lv = 0; int out = 0; int rd = 0; int rv = 0; int w = 0; int x = 0; int y = 0;
    if (blockRadius == nil) { blockRadius = 2; }
    if (maxDisparity == nil) { maxDisparity = 12; }
    w = left[0];
    h = left[1];
    ld = left[3];
    rd = right[3];
    out = arrayx.zerosf(w * h);
    for (y = blockRadius; y < h - blockRadius; ++y) {
        for (x = blockRadius; x < w - blockRadius; ++x) {
            bestD = 0;
            bestCost = 1.0e30;
            dmax = mathx.mini(maxDisparity, x - blockRadius);
            for (disp = 0; disp <= dmax; ++disp) {
                cost = 0.0;
                for (by = 0 - blockRadius; by <= blockRadius; ++by) {
                    lbase = (y + by) * w;
                    for (bx = 0 - blockRadius; bx <= blockRadius; ++bx) {
                        lv = ld[lbase + x + bx];
                        rv = rd[lbase + x + bx - disp];
                        dv = lv - rv;
                        if (dv < 0.0) { dv = 0.0 - dv; }
                        cost = cost + dv;
                    }
                }
                if (cost < bestCost) { bestCost = cost; bestD = disp; }
            }
            out[y * w + x] = bestD * 1.0 / maxDisparity;
        }
    }
    return [w, h, 1, out];
}

# ---------------------------------------------------------------------------
# Point cloud PLY export
# ---------------------------------------------------------------------------

fn save_ply_points(points, colors, path) {
    int c = 0; int cc = 0; int i = 0; int n = 0; int p = 0; int text = 0;
    n = len(points);
    text = "ply\nformat ascii 1.0\nelement vertex {n}\n";
    text = text + "property float x\nproperty float y\nproperty float z\n";
    text = text + "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    for (i = 0; i < n; ++i) {
        p = points[i];
        c = [200, 200, 200];
        if (colors != nil) {
            cc = colors[i];
            c = [int(floor(image.clamp01(cc[0]) * 255.0)),
                 int(floor(image.clamp01(cc[1]) * 255.0)),
                 int(floor(image.clamp01(cc[2]) * 255.0))];
        }
        text = text + strx.fmt_g(p[0], 6) + " " + strx.fmt_g(p[1], 6) + " " +
               strx.fmt_g(p[2], 6) + " " + string(c[0]) + " " + string(c[1]) + " " +
               string(c[2]) + "\n";
    }
    File fh = path;
    fh.write(text);
    return true;
}
