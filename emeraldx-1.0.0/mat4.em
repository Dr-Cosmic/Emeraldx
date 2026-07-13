# =============================================================================
# mat4.em - 4x4 homogeneous transforms (row-major flat arrays of 16 floats)
#
# Conventions: right-handed, column vectors, so a point transforms as
#   p' = M * p    with rows laid out  m[r*4 + c].
# Chained transforms:  world = mul(translate, mul(rotate, scalem))  applies
# scale first, then rotation, then translation.
# =============================================================================

import mathx;
import vector;
import matrix;
# emx-scope-safe

fn identity() {
    return [1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0];
}

fn get(m, r, c) { return m[r * 4 + c]; }
fn set(m, r, c, v) { m[r * 4 + c] = v; return m; }

fn mul(a, b) {
    int c = 0; int k = 0; int out = 0; int r = 0; int s = 0;
    out = [];
    for (r = 0; r < 4; ++r) {
        for (c = 0; c < 4; ++c) {
            s = 0.0;
            for (k = 0; k < 4; ++k) { s = s + a[r * 4 + k] * b[k * 4 + c]; }
            out[r * 4 + c] = s;
        }
    }
    return out;
}

fn transpose(m) {
    int c = 0; int out = 0; int r = 0;
    out = [];
    for (r = 0; r < 4; ++r) {
        for (c = 0; c < 4; ++c) { out[r * 4 + c] = m[c * 4 + r]; }
    }
    return out;
}

fn translation(tx, ty, tz) {
    int m = 0;
    m = identity();
    m[3] = tx; m[7] = ty; m[11] = tz;
    return m;
}

fn scaling(sx, sy, sz) {
    int m = 0;
    if (sy == nil) { sy = sx; }
    if (sz == nil) { sz = sx; }
    m = identity();
    m[0] = sx; m[5] = sy; m[10] = sz;
    return m;
}

fn rot_x(a) {
    int c = 0; int m = 0; int s = 0;
    c = cos(a); s = sin(a);
    m = identity();
    m[5] = c; m[6] = 0.0 - s;
    m[9] = s; m[10] = c;
    return m;
}

fn rot_y(a) {
    int c = 0; int m = 0; int s = 0;
    c = cos(a); s = sin(a);
    m = identity();
    m[0] = c;        m[2] = s;
    m[8] = 0.0 - s;  m[10] = c;
    return m;
}

fn rot_z(a) {
    int c = 0; int m = 0; int s = 0;
    c = cos(a); s = sin(a);
    m = identity();
    m[0] = c; m[1] = 0.0 - s;
    m[4] = s; m[5] = c;
    return m;
}

# Rotation about an arbitrary (unit) axis (Rodrigues form).
fn rot_axis(axis, angle) {
    int c = 0; int s = 0; int t = 0; int u = 0; int x = 0; int y = 0; int z = 0;
    u = vector.vnormalize(axis);
    x = u[0]; y = u[1]; z = u[2];
    c = cos(angle); s = sin(angle); t = 1.0 - c;
    return [t*x*x + c,     t*x*y - s*z,  t*x*z + s*y,  0.0,
            t*x*y + s*z,   t*y*y + c,    t*y*z - s*x,  0.0,
            t*x*z - s*y,   t*y*z + s*x,  t*z*z + c,    0.0,
            0.0,           0.0,          0.0,          1.0];
}

# Compose translate * rotZ * rotY * rotX * scale in one call.
fn trs(t, eulerXYZ, s) {
    int m = 0;
    if (t == nil) { t = [0.0, 0.0, 0.0]; }
    if (eulerXYZ == nil) { eulerXYZ = [0.0, 0.0, 0.0]; }
    if (s == nil) { s = [1.0, 1.0, 1.0]; }
    m = scaling(s[0], s[1], s[2]);
    m = mul(rot_x(eulerXYZ[0]), m);
    m = mul(rot_y(eulerXYZ[1]), m);
    m = mul(rot_z(eulerXYZ[2]), m);
    m = mul(translation(t[0], t[1], t[2]), m);
    return m;
}

# Perspective projection (fovy in degrees). Maps view space (camera at the
# origin looking down -Z) into clip space; NDC z in [-1, 1].
fn perspective(fovyDeg, aspect, near, far) {
    int f = 0; int i = 0; int m = 0;
    f = 1.0 / tan(mathx.rad(fovyDeg) * 0.5);
    m = [];
    for (i = 0; i < 16; ++i) { m[i] = 0.0; }
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (far + near) / (near - far);
    m[11] = (2.0 * far * near) / (near - far);
    m[14] = -1.0;
    return m;
}

fn ortho(l, r, b, t, n, f) {
    int m = 0;
    m = identity();
    m[0] = 2.0 / (r - l);
    m[5] = 2.0 / (t - b);
    m[10] = -2.0 / (f - n);
    m[3] = 0.0 - (r + l) / (r - l);
    m[7] = 0.0 - (t + b) / (t - b);
    m[11] = 0.0 - (f + n) / (f - n);
    return m;
}

# View matrix: camera at eye, looking at target, with the given up hint.
fn look_at(eye, target, up) {
    int f = 0; int s = 0; int u = 0;
    f = vector.vnormalize(vector.vsub(target, eye));    # forward
    s = vector.vnormalize(vector.vcross(f, up));        # right
    u = vector.vcross(s, f);                            # true up
    return [s[0],        s[1],        s[2],        0.0 - vector.vdot(s, eye),
            u[0],        u[1],        u[2],        0.0 - vector.vdot(u, eye),
            0.0 - f[0],  0.0 - f[1],  0.0 - f[2],  vector.vdot(f, eye),
            0.0,         0.0,         0.0,         1.0];
}

# Transform a 3D point (w = 1), returning a 3D point after w-divide.
fn transform_point(m, p) {
    int w = 0; int x = 0; int y = 0; int z = 0;
    x = m[0]*p[0] + m[1]*p[1] + m[2]*p[2] + m[3];
    y = m[4]*p[0] + m[5]*p[1] + m[6]*p[2] + m[7];
    z = m[8]*p[0] + m[9]*p[1] + m[10]*p[2] + m[11];
    w = m[12]*p[0] + m[13]*p[1] + m[14]*p[2] + m[15];
    if (abs(w) < 1.0e-12) { w = 1.0e-12; }
    return [x / w, y / w, z / w];
}

# Transform a direction (w = 0): rotation/scale only, no translation.
fn transform_dir(m, d) {
    return [m[0]*d[0] + m[1]*d[1] + m[2]*d[2],
            m[4]*d[0] + m[5]*d[1] + m[6]*d[2],
            m[8]*d[0] + m[9]*d[1] + m[10]*d[2]];
}

# Full 4-vector transform (no divide). p4 = [x, y, z, w].
fn transform_vec4(m, p4) {
    int k = 0; int out = 0; int r = 0; int s = 0;
    out = [];
    for (r = 0; r < 4; ++r) {
        s = 0.0;
        for (k = 0; k < 4; ++k) { s = s + m[r * 4 + k] * p4[k]; }
        out[r] = s;
    }
    return out;
}

fn to_nested(m) { return matrix.from_flat(m, 4, 4); }
fn from_nested(nm) { return matrix.to_flat(nm); }

fn inverse(m) {
    int inv = 0;
    inv = matrix.inverse(to_nested(m));
    if (inv == nil) { return nil; }
    return from_nested(inv);
}

fn determinant(m) { return matrix.det(to_nested(m)); }

# Inverse-transpose (what normals must be transformed by under non-uniform
# scale). Translation is stripped.
fn normal_matrix(m) {
    int n = 0;
    n = inverse(m);
    if (n == nil) { return identity(); }
    n = transpose(n);
    n[3] = 0.0; n[7] = 0.0; n[11] = 0.0;
    n[12] = 0.0; n[13] = 0.0; n[14] = 0.0; n[15] = 1.0;
    return n;
}
