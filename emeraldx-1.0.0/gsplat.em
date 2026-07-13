# =============================================================================
# gsplat.em - 3D Gaussian Splatting renderer
#
# Each splat is an anisotropic 3D gaussian:
#     g = [pos(3), scale(3), rotQuat(4), color(3), opacity]
# Its covariance is built exactly as in the 3DGS paper:
#     Sigma = R S S^T R^T          (R from the quaternion, S = diag(scale))
# Rendering projects each gaussian to a 2D gaussian with the EWA Jacobian:
#     Sigma' = J W Sigma W^T J^T   (W = camera rotation, J = perspective Jac.)
# and alpha-composites splats front to back:
#     C += T * alpha_i * c_i;   T *= (1 - alpha_i)
#
#   splats = [g0, g1, ...]
#   img = gsplat.render(splats, cam, w, h, bg)
#   cam = gsplat.make_camera(eye, target, up, fovDeg, w, h)
#   gsplat.save_ply(splats, path) / load_ply(path)
#   gsplat.from_points(points, colors, size)  - init from a point cloud
# =============================================================================

import mathx;
import arrayx;
import vector;
import matrix;
import quat;
import mesh;
import strx;
import image;
# emx-scope-safe

fn make_splat(pos, scale, rotQ, color, opacity) {
    return [pos, scale, rotQ, color, opacity];
}

# 3x3 covariance (nested rows) from scale + rotation quaternion.
fn covariance3d(scale, rotQ) {
    int i = 0; int m = 0; int r = 0; int row = 0;
    r = quat.to_mat3(rotQ);
    # M = R * S  (columns scaled)
    m = [];
    for (i = 0; i < 3; ++i) {
        row = r[i];
        m[i] = [row[0] * scale[0], row[1] * scale[1], row[2] * scale[2]];
    }
    return matrix.matmul(m, matrix.transpose(m));
}

# Camera = [viewMat4, fx, fy, w, h]. fx/fy in pixels.
fn make_camera(eye, target, up, fovDeg, w, h) {
    int fy = 0; int view = 0;
    view = look_view(eye, target, up);
    fy = 0.5 * h / tan(mathx.rad(fovDeg) * 0.5);
    return [view, fy, fy, w, h];
}

# Row-major 4x4 view matrix (world -> camera, camera looks down -z).
fn look_view(eye, target, up) {
    int f = 0; int s = 0; int u = 0;
    f = vector.vnormalize(vector.vsub(target, eye));
    s = vector.vnormalize(vector.vcross(f, up));
    u = vector.vcross(s, f);
    return [s[0], s[1], s[2], 0.0 - vector.vdot(s, eye),
            u[0], u[1], u[2], 0.0 - vector.vdot(u, eye),
            0.0 - f[0], 0.0 - f[1], 0.0 - f[2], vector.vdot(f, eye),
            0.0, 0.0, 0.0, 1.0];
}

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

fn render(splats, cam, bg) {
    int a = 0; int alpha = 0; int bq = 0; int c = 0; int col = 0; int cov2 = 0; int cx = 0; int cy = 0; int d = 0; int depth = 0; int depths = 0; int det = 0; int disc = 0; int drawable = 0; int dx = 0; int dy = 0; int fx = 0; int fy = 0; int g = 0; int h = 0; int i = 0; int ia = 0; int ib = 0; int ic = 0; int inv = 0; int invD = 0; int invD2 = 0; int j = 0; int jw = 0; int jwm = 0; int k = 0; int lam = 0; int mid = 0; int n = 0; int nd = 0; int oi = 0; int opac = 0; int order = 0; int p = 0; int pix = 0; int power = 0; int px2 = 0; int py = 0; int radius = 0; int sigma = 0; int t = 0; int trans = 0; int tx = 0; int ty = 0; int tz = 0; int view = 0; int w = 0; int wgt = 0; int wr = 0; int x0 = 0; int x1 = 0; int y0 = 0; int y1 = 0;
    view = cam[0];
    fx = cam[1];
    fy = cam[2];
    w = cam[3];
    h = cam[4];
    if (bg == nil) { bg = [0.0, 0.0, 0.0]; }
    n = len(splats);

    # camera rotation W as nested 3x3 (upper-left of view)
    wr = [[view[0], view[1], view[2]],
          [view[4], view[5], view[6]],
          [view[8], view[9], view[10]]];

    # project every splat: keep [depth, cx, cy, inv2x2(a,b,c), color, opac]
    drawable = [];
    depths = [];
    for (k = 0; k < n; ++k) {
        g = splats[k];
        p = g[0];
        # camera-space center
        tx = view[0] * p[0] + view[1] * p[1] + view[2] * p[2] + view[3];
        ty = view[4] * p[0] + view[5] * p[1] + view[6] * p[2] + view[7];
        tz = view[8] * p[0] + view[9] * p[1] + view[10] * p[2] + view[11];
        if (tz >= -0.05) { continue; }              # behind / too close
        depth = 0.0 - tz;
        # screen position (pixels)
        cx = fx * tx / depth + w * 0.5;
        cy = 0.0 - fy * ty / depth + h * 0.5;
        # EWA: Sigma' = J W Sigma W^T J^T
        sigma = covariance3d(g[1], g[2]);
        jw = [];
        # J = [[fx/d, 0, fx*tx/d^2], [0, fy/d, fy*ty/d^2]] in -z convention:
        # using depth d = -tz and camera looks down -z, the sign folds into tz
        invD = 1.0 / depth;
        invD2 = invD * invD;
        j = [[fx * invD, 0.0, fx * tx * invD2],
             [0.0, fy * invD, fy * ty * invD2]];
        jwm = matrix.matmul(j, wr);
        cov2 = matrix.matmul(jwm, matrix.matmul(sigma, matrix.transpose(jwm)));
        a = cov2[0][0] + 0.3;                       # low-pass (paper's +0.3)
        bq = cov2[0][1];
        c = cov2[1][1] + 0.3;
        det = a * c - bq * bq;
        if (det < 1.0e-9) { continue; }
        inv = 1.0 / det;
        drawable[len(drawable)] = [depth, cx, cy, c * inv, 0.0 - bq * inv,
                                   a * inv, g[3], g[4], a, bq, c];
        depths[len(depths)] = depth;
    }

    # front-to-back order
    order = arrayx.argsort(depths);

    pix = [];
    for (i = 0; i < w * h; ++i) {
        pix[i * 3] = 0.0;
        pix[i * 3 + 1] = 0.0;
        pix[i * 3 + 2] = 0.0;
    }
    trans = arrayx.full(w * h, 1.0);

    nd = len(drawable);
    for (oi = 0; oi < nd; ++oi) {
        d = drawable[order[oi]];
        cx = d[1];
        cy = d[2];
        ia = d[3];
        ib = d[4];
        ic = d[5];
        col = d[6];
        opac = d[7];
        # 3-sigma bounding radius from the covariance eigen bound
        mid = (d[8] + d[10]) * 0.5;
        disc = mid * mid - (d[8] * d[10] - d[9] * d[9]);
        if (disc < 0.0) { disc = 0.0; }
        lam = mid + sqrt(disc);
        radius = 3.0 * sqrt(mathx.maxv(lam, 0.01));
        x0 = mathx.maxi(int(floor(cx - radius)), 0);
        x1 = mathx.mini(int(top(cx + radius)), w - 1);
        y0 = mathx.maxi(int(floor(cy - radius)), 0);
        y1 = mathx.mini(int(top(cy + radius)), h - 1);
        for (py = y0; py <= y1; ++py) {
            for (px2 = x0; px2 <= x1; ++px2) {
                i = py * w + px2;
                t = trans[i];
                if (t < 0.004) { continue; }        # saturated
                dx = px2 + 0.5 - cx;
                dy = py + 0.5 - cy;
                power = 0.0 - 0.5 * (ia * dx * dx + 2.0 * ib * dx * dy + ic * dy * dy);
                if (power < -12.0) { continue; }
                alpha = opac * exp(power);
                if (alpha < 0.004) { continue; }
                if (alpha > 0.99) { alpha = 0.99; }
                wgt = t * alpha;
                pix[i * 3] = pix[i * 3] + wgt * col[0];
                pix[i * 3 + 1] = pix[i * 3 + 1] + wgt * col[1];
                pix[i * 3 + 2] = pix[i * 3 + 2] + wgt * col[2];
                trans[i] = t * (1.0 - alpha);
            }
        }
    }
    # composite over background
    for (i = 0; i < w * h; ++i) {
        t = trans[i];
        pix[i * 3] = image.clamp01(pix[i * 3] + t * bg[0]);
        pix[i * 3 + 1] = image.clamp01(pix[i * 3 + 1] + t * bg[1]);
        pix[i * 3 + 2] = image.clamp01(pix[i * 3 + 2] + t * bg[2]);
    }
    return [w, h, 3, pix];
}

# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

# Isotropic splats from a point cloud. points: array of 3-vectors; colors:
# array of rgb (or nil for position-derived); size: gaussian radius.
fn from_points(points, colors, size, opacity) {
    int c = 0; int i = 0; int idq = 0; int out = 0; int p = 0;
    if (opacity == nil) { opacity = 0.8; }
    out = [];
    idq = quat.identity();
    for (i = 0; i < len(points); ++i) {
        p = points[i];
        c = nil;
        if (colors != nil) { c = colors[i]; }
        if (c == nil) {
            c = [mathx.clampf(0.5 + p[0], 0.0, 1.0),
                 mathx.clampf(0.5 + p[1], 0.0, 1.0),
                 mathx.clampf(0.5 + p[2], 0.0, 1.0)];
        }
        out[i] = [p, [size, size, size], idq, c, opacity];
    }
    return out;
}

# Splats sampled over a mesh surface (one per face centroid), oriented by the
# face normal and sized by the face area - a quick "splatify" for models.
fn from_mesh(m, opacity) {
    int a = 0; int area = 0; int b = 0; int c = 0; int centroid = 0; int col = 0; int e1 = 0; int e2 = 0; int f = 0; int faces = 0; int ia = 0; int ib = 0; int ic = 0; int nf = 0; int nrm = 0; int out = 0; int positions = 0; int q = 0; int r = 0;
    if (opacity == nil) { opacity = 0.85; }
    positions = m[0];
    faces = m[3];
    nf = len(faces) // 3;
    out = [];
    for (f = 0; f < nf; ++f) {
        ia = faces[f * 3];
        ib = faces[f * 3 + 1];
        ic = faces[f * 3 + 2];
        a = mesh.get_position(m, ia);
        b = mesh.get_position(m, ib);
        c = mesh.get_position(m, ic);
        centroid = [(a[0] + b[0] + c[0]) / 3.0,
                    (a[1] + b[1] + c[1]) / 3.0,
                    (a[2] + b[2] + c[2]) / 3.0];
        e1 = vector.vsub(b, a);
        e2 = vector.vsub(c, a);
        nrm = vector.vcross(e1, e2);
        area = vector.vnorm(nrm) * 0.5;
        r = sqrt(area) * 0.7;
        # orientation: rotate z axis onto the normal, flatten along it
        q = quat.from_to([0.0, 0.0, 1.0], vector.vnormalize(nrm));
        col = [mathx.clampf(0.5 + centroid[0] * 0.6, 0.0, 1.0),
               mathx.clampf(0.5 + centroid[1] * 0.6, 0.0, 1.0),
               mathx.clampf(0.5 + centroid[2] * 0.6, 0.0, 1.0)];
        out[f] = [centroid, [r, r, r * 0.25], q, col, opacity];
    }
    return out;
}

# ---------------------------------------------------------------------------
# ASCII PLY I/O (3DGS-flavored property names)
# ---------------------------------------------------------------------------

fn save_ply(splats, path) {
    int c = 0; int g = 0; int i = 0; int k = 0; int line = 0; int n = 0; int p = 0; int q = 0; int s = 0; int text = 0; int vals = 0;
    n = len(splats);
    text = "ply\nformat ascii 1.0\nelement vertex {n}\n";
    text = text + "property float x\nproperty float y\nproperty float z\n";
    text = text + "property float scale_0\nproperty float scale_1\nproperty float scale_2\n";
    text = text + "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n";
    text = text + "property float red\nproperty float green\nproperty float blue\n";
    text = text + "property float opacity\nend_header\n";
    for (i = 0; i < n; ++i) {
        g = splats[i];
        p = g[0];
        s = g[1];
        q = g[2];
        c = g[3];
        vals = [p[0], p[1], p[2], s[0], s[1], s[2],
                q[0], q[1], q[2], q[3], c[0], c[1], c[2], g[4]];
        line = "";
        for (k = 0; k < len(vals); ++k) {
            line = line + strx.fmt_g(vals[k], 6);
            if (k < len(vals) - 1) { line = line + " "; }
        }
        text = text + line + "\n";
    }
    File fh = path;
    fh.write(text);
    return true;
}

fn load_ply(path) {
    int count = 0; int dataStart = 0; int i = 0; int k = 0; int lines = 0; int out = 0; int parts = 0; int t = 0; int text = 0; int v = 0;
    File fh = path;
    text = fh.read();
    if (text == nil || len(text) == 0) { return nil; }
    lines = strx.split_lines(text);
    count = 0;
    dataStart = -1;
    for (i = 0; i < len(lines); ++i) {
        t = strx.trim(lines[i]);
        parts = strx.split_ws(t);
        if (len(parts) >= 3 && parts[0] == "element" && parts[1] == "vertex") {
            count = strx.parse_int(parts[2]);
        }
        if (t == "end_header") { dataStart = i + 1; break; }
    }
    if (dataStart < 0) { return nil; }
    out = [];
    for (i = 0; i < count; ++i) {
        parts = strx.split_ws(strx.trim(lines[dataStart + i]));
        v = [];
        for (k = 0; k < len(parts); ++k) { v[k] = strx.parse_float(parts[k]); }
        out[i] = [[v[0], v[1], v[2]], [v[3], v[4], v[5]],
                  [v[6], v[7], v[8], v[9]], [v[10], v[11], v[12]], v[13]];
    }
    return out;
}
