# =============================================================================
# vector.em - n-dimensional vectors as plain arrays
# =============================================================================

import mathx;
# emx-scope-safe

fn vec2(x, y) { return [x, y]; }
fn vec3(x, y, z) { return [x, y, z]; }
fn vec4(x, y, z, w) { return [x, y, z, w]; }

fn vzeros(n) {
    int i = 0; int v = 0;
    v = [];
    for (i = 0; i < n; ++i) { v[i] = 0.0; }
    return v;
}

fn vfull(n, x) {
    int i = 0; int v = 0;
    v = [];
    for (i = 0; i < n; ++i) { v[i] = x; }
    return v;
}

fn vadd(a, b) {
    int i = 0; int out = 0;
    out = [];
    for (i = 0; i < len(a); ++i) { out[i] = a[i] + b[i]; }
    return out;
}

fn vsub(a, b) {
    int i = 0; int out = 0;
    out = [];
    for (i = 0; i < len(a); ++i) { out[i] = a[i] - b[i]; }
    return out;
}

fn vscale(a, s) {
    int i = 0; int out = 0;
    out = [];
    for (i = 0; i < len(a); ++i) { out[i] = a[i] * s; }
    return out;
}

fn vneg(a) { return vscale(a, -1.0); }

fn vhadamard(a, b) {
    int i = 0; int out = 0;
    out = [];
    for (i = 0; i < len(a); ++i) { out[i] = a[i] * b[i]; }
    return out;
}

# a + b*s in one pass (very common in physics integrators).
fn vaxpy(a, b, s) {
    int i = 0; int out = 0;
    out = [];
    for (i = 0; i < len(a); ++i) { out[i] = a[i] + b[i] * s; }
    return out;
}

fn vdot(a, b) {
    int i = 0; int s = 0;
    s = 0.0;
    for (i = 0; i < len(a); ++i) { s = s + a[i] * b[i]; }
    return s;
}

fn vnorm_sq(a) { return vdot(a, a); }
fn vnorm(a) { return sqrt(vdot(a, a)); }

fn vdist(a, b) { return vnorm(vsub(a, b)); }
fn vdist_sq(a, b) { return vnorm_sq(vsub(a, b)); }

fn vnormalize(a) {
    int n = 0;
    n = vnorm(a);
    if (n < 1.0e-12) { return vzeros(len(a)); }
    return vscale(a, 1.0 / n);
}

fn vlerp(a, b, t) {
    int i = 0; int out = 0;
    out = [];
    for (i = 0; i < len(a); ++i) { out[i] = a[i] + (b[i] - a[i]) * t; }
    return out;
}

fn vcross(a, b) {
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]];
}

fn vangle(a, b) {
    int d = 0;
    d = vnorm(a) * vnorm(b);
    if (d < 1.0e-12) { return 0.0; }
    return acos(mathx.clampf(vdot(a, b) / d, -1.0, 1.0));
}

# Projection of a onto b, and the remainder perpendicular to b.
fn vproject(a, b) {
    int d = 0;
    d = vnorm_sq(b);
    if (d < 1.0e-12) { return vzeros(len(a)); }
    return vscale(b, vdot(a, b) / d);
}

fn vreject(a, b) { return vsub(a, vproject(a, b)); }

# Reflect v about unit normal n:  v - 2 (v.n) n
fn vreflect(v, n) { return vsub(v, vscale(n, 2.0 * vdot(v, n))); }

fn vclamp_norm(v, maxLen) {
    int n = 0;
    n = vnorm(v);
    if (n <= maxLen || n < 1.0e-12) { return v; }
    return vscale(v, maxLen / n);
}

fn vabs(a) {
    int i = 0; int out = 0;
    out = [];
    for (i = 0; i < len(a); ++i) { out[i] = abs(a[i]); }
    return out;
}

fn vmin(a, b) {
    int i = 0; int out = 0;
    out = [];
    for (i = 0; i < len(a); ++i) { out[i] = mathx.minv(a[i], b[i]); }
    return out;
}

fn vmax(a, b) {
    int i = 0; int out = 0;
    out = [];
    for (i = 0; i < len(a); ++i) { out[i] = mathx.maxv(a[i], b[i]); }
    return out;
}

fn vsum(a) {
    int s = 0;
    s = 0.0;
    for x in a { s = s + x; }
    return s;
}

fn vmean(a) {
    if (len(a) == 0) { return 0.0; }
    return vsum(a) / len(a);
}

# Any vector perpendicular to a 3D vector (unit length).
fn vperp3(a) {
    if (abs(a[0]) < 0.9) { return vnormalize(vcross(a, [1.0, 0.0, 0.0])); }
    return vnormalize(vcross(a, [0.0, 1.0, 0.0]));
}

# Spherical linear interpolation between two (roughly unit) vectors.
fn vslerp(a, b, t) {
    int d = 0; int s = 0; int th = 0;
    d = mathx.clampf(vdot(vnormalize(a), vnormalize(b)), -1.0, 1.0);
    th = acos(d);
    if (abs(th) < 1.0e-6) { return vlerp(a, b, t); }
    s = sin(th);
    return vadd(vscale(a, sin((1.0 - t) * th) / s), vscale(b, sin(t * th) / s));
}
