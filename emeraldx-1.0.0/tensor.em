# =============================================================================
# tensor.em - N-dimensional tensors for Emerald
#
# A tensor is t = [shape, data]: shape is an int array, data is a flat
# row-major float array. For hot loops, pull the data out once, mutate the
# local copy in place, and wrap it back up:
#
#     d = t[1];                  # one copy out
#     d[k] = 5.0;                # in-place on the local array
#     t = [t[0], d];             # one copy back
# =============================================================================

import mathx;
import arrayx;
import prng;
import matrix;
# emx-scope-safe

fn tsize(shape) {
    int n = 0;
    n = 1;
    for d in shape { n = n * d; }
    return n;
}

fn tnew(shape, fill) {
    if (fill == nil) { fill = 0.0; }
    return [shape, arrayx.full(tsize(shape), fill)];
}

fn tzeros(shape) { return tnew(shape, 0.0); }
fn tones(shape) { return tnew(shape, 1.0); }

fn from_flat(shape, data) { return [shape, data]; }

fn shape(t) { return t[0]; }
fn data(t) { return t[1]; }
fn ndim(t) { return len(t[0]); }
fn numel(t) { return tsize(t[0]); }

# Row-major strides for a shape.
fn strides(shp) {
    int i = 0; int n = 0; int st = 0;
    n = len(shp);
    st = [];
    for (i = 0; i < n; ++i) { st[i] = 1; }
    for (i = n - 2; i >= 0; --i) { st[i] = st[i + 1] * shp[i + 1]; }
    return st;
}

# Flat offset of a multi-index.
fn offset(shp, idx) {
    int i = 0; int k = 0; int st = 0;
    st = strides(shp);
    k = 0;
    for (i = 0; i < len(idx); ++i) { k = k + idx[i] * st[i]; }
    return k;
}

fn tget(t, idx) {
    int d = 0;
    d = t[1];
    return d[offset(t[0], idx)];
}

# Returns a new tensor (value semantics). For bulk writes see header note.
fn tset(t, idx, v) {
    int d = 0;
    d = t[1];
    d[offset(t[0], idx)] = v;
    return [t[0], d];
}

fn reshape(t, newShape) {
    return [newShape, t[1]];
}

fn trand(shape, seed, lo, hi) {
    int r = 0;
    r = prng.randf_array(seed, tsize(shape), lo, hi);
    return [[shape, r[0]], r[1]];
}

fn trandn(shape, seed, scale) {
    int d = 0; int i = 0; int r = 0;
    if (scale == nil) { scale = 1.0; }
    r = prng.normal_array(seed, tsize(shape));
    d = r[0];
    for (i = 0; i < len(d); ++i) { d[i] = d[i] * scale; }
    return [[shape, d], r[1]];
}

# ---------------------------------------------------------------------------
# Elementwise operations
# ---------------------------------------------------------------------------

fn tadd(a, b) {
    int da = 0; int db = 0; int i = 0; int out = 0;
    da = a[1]; db = b[1];
    out = [];
    for (i = 0; i < len(da); ++i) { out[i] = da[i] + db[i]; }
    return [a[0], out];
}

fn tsub(a, b) {
    int da = 0; int db = 0; int i = 0; int out = 0;
    da = a[1]; db = b[1];
    out = [];
    for (i = 0; i < len(da); ++i) { out[i] = da[i] - db[i]; }
    return [a[0], out];
}

fn tmul(a, b) {
    int da = 0; int db = 0; int i = 0; int out = 0;
    da = a[1]; db = b[1];
    out = [];
    for (i = 0; i < len(da); ++i) { out[i] = da[i] * db[i]; }
    return [a[0], out];
}

fn tscale(a, s) {
    int da = 0; int i = 0; int out = 0;
    da = a[1];
    out = [];
    for (i = 0; i < len(da); ++i) { out[i] = da[i] * s; }
    return [a[0], out];
}

fn tadd_scalar(a, s) {
    int da = 0; int i = 0; int out = 0;
    da = a[1];
    out = [];
    for (i = 0; i < len(da); ++i) { out[i] = da[i] + s; }
    return [a[0], out];
}

fn tapply(a, f) {
    int da = 0; int i = 0; int out = 0;
    da = a[1];
    out = [];
    for (i = 0; i < len(da); ++i) { out[i] = f(da[i]); }
    return [a[0], out];
}

fn tsum(a) {
    int s = 0;
    s = 0.0;
    for x in a[1] { s = s + x; }
    return s;
}

fn tmean(a) {
    int n = 0;
    n = numel(a);
    if (n == 0) { return 0.0; }
    return tsum(a) / n;
}

fn tmin(a) { return arrayx.minv(a[1]); }
fn tmax(a) { return arrayx.maxv(a[1]); }
fn targmax(a) { return arrayx.argmax(a[1]); }

# ---------------------------------------------------------------------------
# Axis operations (general rank)
# ---------------------------------------------------------------------------

# Sum over one axis; the result drops that axis.
fn tsum_axis(t, axis) {
    int axN = 0; int axStride = 0; int baseIn = 0; int baseOut = 0; int d = 0; int i = 0; int inner = 0; int k = 0; int nd = 0; int o = 0; int out = 0; int outShape = 0; int outer = 0; int rowBase = 0; int shp = 0; int st = 0; int total = 0;
    shp = t[0];
    d = t[1];
    nd = len(shp);
    outShape = [];
    for (i = 0; i < nd; ++i) { if (i != axis) { outShape[len(outShape)] = shp[i]; } }
    if (len(outShape) == 0) { outShape = [1]; }
    out = arrayx.zerosf(tsize(outShape));
    st = strides(shp);
    axN = shp[axis];
    axStride = st[axis];
    total = len(d);
    outer = total // (axN * axStride);
    for (o = 0; o < outer; ++o) {
        baseOut = o * axStride;
        baseIn = o * axN * axStride;
        for (k = 0; k < axN; ++k) {
            rowBase = baseIn + k * axStride;
            for (inner = 0; inner < axStride; ++inner) {
                out[baseOut + inner] = out[baseOut + inner] + d[rowBase + inner];
            }
        }
    }
    return [outShape, out];
}

fn tmean_axis(t, axis) {
    int n = 0; int r = 0;
    r = tsum_axis(t, axis);
    n = t[0];
    return tscale(r, 1.0 / n[axis]);
}

# Slice index `i` along axis 0 (e.g. one sample of a batch).
fn tslice0(t, i) {
    int base = 0; int d = 0; int k = 0; int n = 0; int out = 0; int shp = 0; int sub = 0;
    shp = t[0];
    d = t[1];
    sub = [];
    for (k = 1; k < len(shp); ++k) { sub[len(sub)] = shp[k]; }
    if (len(sub) == 0) { sub = [1]; }
    n = tsize(sub);
    out = [];
    base = i * n;
    for (k = 0; k < n; ++k) { out[k] = d[base + k]; }
    return [sub, out];
}

# Stack same-shaped tensors along a new leading axis.
fn tstack(list) {
    int d = 0; int first = 0; int out = 0; int outShape = 0; int shp = 0;
    first = list[0];
    shp = first[0];
    outShape = [len(list)];
    for x in shp { outShape[len(outShape)] = x; }
    out = [];
    for t in list {
        d = t[1];
        for x in d { out[len(out)] = x; }
    }
    return [outShape, out];
}

# ---------------------------------------------------------------------------
# 2D tensor (matrix) helpers
# ---------------------------------------------------------------------------

fn tmatmul(a, b) {
    int aik = 0; int da = 0; int db = 0; int i = 0; int j = 0; int k = 0; int m = 0; int n = 0; int out = 0; int p = 0; int sa = 0; int sb = 0;
    sa = a[0]; sb = b[0];
    n = sa[0]; m = sa[1]; p = sb[1];
    da = a[1]; db = b[1];
    out = arrayx.zerosf(n * p);
    for (i = 0; i < n; ++i) {
        for (k = 0; k < m; ++k) {
            aik = da[i * m + k];
            if (aik != 0.0) {
                for (j = 0; j < p; ++j) {
                    out[i * p + j] = out[i * p + j] + aik * db[k * p + j];
                }
            }
        }
    }
    return [[n, p], out];
}

fn ttranspose2d(t) {
    int c = 0; int d = 0; int i = 0; int j = 0; int out = 0; int r = 0; int shp = 0;
    shp = t[0];
    r = shp[0]; c = shp[1];
    d = t[1];
    out = [];
    for (j = 0; j < c; ++j) {
        for (i = 0; i < r; ++i) { out[j * r + i] = d[i * c + j]; }
    }
    return [[c, r], out];
}

# Add a length-c row vector to every row of an (r x c) tensor.
fn tadd_rowvec(t, v) {
    int c = 0; int d = 0; int i = 0; int j = 0; int out = 0; int r = 0; int shp = 0;
    shp = t[0];
    r = shp[0]; c = shp[1];
    d = t[1];
    out = [];
    for (i = 0; i < r; ++i) {
        for (j = 0; j < c; ++j) { out[i * c + j] = d[i * c + j] + v[j]; }
    }
    return [[r, c], out];
}

# Row-wise softmax of a 2D tensor (numerically stabilized).
fn tsoftmax_rows(t) {
    int base = 0; int c = 0; int d = 0; int i = 0; int j = 0; int m = 0; int out = 0; int r = 0; int s = 0; int shp = 0; int v = 0;
    shp = t[0];
    r = shp[0]; c = shp[1];
    d = t[1];
    out = [];
    for (i = 0; i < r; ++i) {
        base = i * c;
        m = d[base];
        for (j = 1; j < c; ++j) { if (d[base + j] > m) { m = d[base + j]; } }
        s = 0.0;
        for (j = 0; j < c; ++j) {
            v = exp(d[base + j] - m);
            out[base + j] = v;
            s = s + v;
        }
        for (j = 0; j < c; ++j) { out[base + j] = out[base + j] / s; }
    }
    return [[r, c], out];
}

fn to_matrix(t) {
    int c = 0; int d = 0; int i = 0; int j = 0; int m = 0; int r = 0; int row = 0; int shp = 0;
    shp = t[0];
    r = shp[0]; c = shp[1];
    d = t[1];
    m = [];
    for (i = 0; i < r; ++i) {
        row = [];
        for (j = 0; j < c; ++j) { row[j] = d[i * c + j]; }
        m[i] = row;
    }
    return m;
}

fn from_matrix(m) {
    int c = 0; int i = 0; int j = 0; int out = 0; int r = 0; int row = 0;
    r = len(m);
    c = 0;
    if (r > 0) { c = len(m[0]); }
    out = [];
    for (i = 0; i < r; ++i) {
        row = m[i];
        for (j = 0; j < c; ++j) { out[len(out)] = row[j]; }
    }
    return [[r, c], out];
}

fn tprint(t, decimals) {
    int m = 0; int shp = 0;
    shp = t[0];
    print("tensor shape ", shp);
    if (len(shp) == 2) {
        m = to_matrix(t);
        matrix.print_mat(m, decimals);
    } else {
        print(t[1]);
    }
}
