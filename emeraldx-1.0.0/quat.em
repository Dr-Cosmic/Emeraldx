# =============================================================================
# quat.em - unit quaternions for 3D rotation, stored as [w, x, y, z]
# =============================================================================

import mathx;
import vector;
# emx-scope-safe

fn identity() { return [1.0, 0.0, 0.0, 0.0]; }

fn from_axis_angle(axis, angle) {
    int h = 0; int s = 0; int u = 0;
    u = vector.vnormalize(axis);
    h = angle * 0.5;
    s = sin(h);
    return [cos(h), u[0] * s, u[1] * s, u[2] * s];
}

# Intrinsic XYZ euler angles (radians) to quaternion.
fn from_euler(rx, ry, rz) {
    int cx = 0; int cy = 0; int cz = 0; int sx = 0; int sy = 0; int sz = 0;
    cx = cos(rx * 0.5); sx = sin(rx * 0.5);
    cy = cos(ry * 0.5); sy = sin(ry * 0.5);
    cz = cos(rz * 0.5); sz = sin(rz * 0.5);
    return [cx*cy*cz + sx*sy*sz,
            sx*cy*cz - cx*sy*sz,
            cx*sy*cz + sx*cy*sz,
            cx*cy*sz - sx*sy*cz];
}

fn qmul(a, b) {
    return [a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
            a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
            a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
            a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]];
}

fn conj(q) { return [q[0], 0.0 - q[1], 0.0 - q[2], 0.0 - q[3]]; }

fn qdot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]; }

fn qnorm(q) { return sqrt(qdot(q, q)); }

fn normalize(q) {
    int n = 0;
    n = qnorm(q);
    if (n < 1.0e-12) { return identity(); }
    return [q[0]/n, q[1]/n, q[2]/n, q[3]/n];
}

# Rotate a 3D vector: v' = q v q*  (q assumed unit).
fn rotate_vec(q, v) {
    int qv = 0; int uuv = 0; int uv = 0;
    qv = [q[1], q[2], q[3]];
    uv = vector.vcross(qv, v);
    uuv = vector.vcross(qv, uv);
    return vector.vadd(v, vector.vadd(vector.vscale(uv, 2.0 * q[0]),
                                      vector.vscale(uuv, 2.0)));
}

# Row-major 4x4 rotation matrix for a unit quaternion (matches mat4.em).
fn to_mat4(q) {
    int w = 0; int x = 0; int y = 0; int z = 0;
    w = q[0]; x = q[1]; y = q[2]; z = q[3];
    return [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z),       2.0*(x*z + w*y),       0.0,
            2.0*(x*y + w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x),       0.0,
            2.0*(x*z - w*y),       2.0*(y*z + w*x),       1.0 - 2.0*(x*x + y*y), 0.0,
            0.0,                   0.0,                   0.0,                   1.0];
}

# 3x3 rotation matrix as a nested array (for covariance math etc).
fn to_mat3(q) {
    int m = 0; int w = 0; int x = 0; int y = 0; int z = 0;
    w = q[0]; x = q[1]; y = q[2]; z = q[3];
    m = [];
    m[0] = [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z),       2.0*(x*z + w*y)];
    m[1] = [2.0*(x*y + w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x)];
    m[2] = [2.0*(x*z - w*y),       2.0*(y*z + w*x),       1.0 - 2.0*(x*x + y*y)];
    return m;
}

# Normalized linear interpolation (cheap, fine for small angles).
fn nlerp(a, b, t) {
    if (qdot(a, b) < 0) { b = [0.0 - b[0], 0.0 - b[1], 0.0 - b[2], 0.0 - b[3]]; }
    return normalize([a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t,
                      a[2] + (b[2]-a[2])*t, a[3] + (b[3]-a[3])*t]);
}

# Spherical linear interpolation.
fn slerp(a, b, t) {
    int d = 0; int s = 0; int th = 0; int wa = 0; int wb = 0;
    d = qdot(a, b);
    if (d < 0) {
        b = [0.0 - b[0], 0.0 - b[1], 0.0 - b[2], 0.0 - b[3]];
        d = 0.0 - d;
    }
    if (d > 0.9995) { return nlerp(a, b, t); }
    th = acos(mathx.clampf(d, -1.0, 1.0));
    s = sin(th);
    wa = sin((1.0 - t) * th) / s;
    wb = sin(t * th) / s;
    return [a[0]*wa + b[0]*wb, a[1]*wa + b[1]*wb,
            a[2]*wa + b[2]*wb, a[3]*wa + b[3]*wb];
}

# Shortest-arc rotation taking unit vector `fromV` to unit vector `toV`.
fn from_to(fromV, toV) {
    int ax = 0; int d = 0; int q = 0;
    d = vector.vdot(fromV, toV);
    if (d > 0.999999) { return identity(); }
    if (d < -0.999999) {
        ax = vector.vperp3(fromV);
        return from_axis_angle(ax, pi);
    }
    ax = vector.vcross(fromV, toV);
    q = [1.0 + d, ax[0], ax[1], ax[2]];
    return normalize(q);
}
