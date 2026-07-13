# =============================================================================
# matrix.em - dense matrices for Emerald
#
# A matrix is an array of row arrays: m[i][j]. All operations are functional:
# they build and return new matrices. Decompositions:
#   lu()          - LU with partial pivoting  -> det, solve, inverse
#   qr()          - modified Gram-Schmidt     -> least squares
#   eig_jacobi()  - symmetric eigen           -> PCA, spectral methods
#   svd()         - via eigen of A^T A        -> pinv, null vectors, DLT
# =============================================================================

import mathx;
import prng;
import strx;
# emx-scope-safe

fn rows(m) { return len(m); }
fn cols(m) {
    if (len(m) == 0) { return 0; }
    return len(m[0]);
}

fn mnew(r, c, fill) {
    int i = 0; int j = 0; int m = 0; int row = 0;
    if (fill == nil) { fill = 0.0; }
    m = [];
    for (i = 0; i < r; ++i) {
        row = [];
        for (j = 0; j < c; ++j) { row[j] = fill; }
        m[i] = row;
    }
    return m;
}

fn zeros(r, c) { return mnew(r, c, 0.0); }
fn ones(r, c) { return mnew(r, c, 1.0); }

fn identity(n) {
    int i = 0; int m = 0;
    m = mnew(n, n, 0.0);
    for (i = 0; i < n; ++i) { m[i][i] = 1.0; }
    return m;
}

fn diag(v) {
    int i = 0; int m = 0; int n = 0;
    n = len(v);
    m = mnew(n, n, 0.0);
    for (i = 0; i < n; ++i) { m[i][i] = v[i]; }
    return m;
}

fn get_diag(m) {
    int i = 0; int n = 0; int v = 0;
    n = mathx.minv(rows(m), cols(m));
    v = [];
    for (i = 0; i < n; ++i) { v[i] = m[i][i]; }
    return v;
}

# Build an r x c matrix from a flat row-major array.
fn from_flat(flat, r, c) {
    int i = 0; int j = 0; int k = 0; int m = 0; int row = 0;
    m = [];
    k = 0;
    for (i = 0; i < r; ++i) {
        row = [];
        for (j = 0; j < c; ++j) { row[j] = flat[k]; k = k + 1; }
        m[i] = row;
    }
    return m;
}

fn to_flat(m) {
    int i = 0; int j = 0; int out = 0; int row = 0;
    out = [];
    for (i = 0; i < rows(m); ++i) {
        row = m[i];
        for (j = 0; j < len(row); ++j) { out[len(out)] = row[j]; }
    }
    return out;
}

# Uniform random matrix in [lo, hi). Returns [m, newSeed].
fn rand(r, c, seed, lo, hi) {
    int i = 0; int j = 0; int m = 0; int row = 0; int rr = 0;
    if (lo == nil) { lo = 0.0; }
    if (hi == nil) { hi = 1.0; }
    m = [];
    for (i = 0; i < r; ++i) {
        row = [];
        for (j = 0; j < c; ++j) {
            rr = prng.randf(seed);
            row[j] = lo + (hi - lo) * rr[0];
            seed = rr[1];
        }
        m[i] = row;
    }
    return [m, seed];
}

# Gaussian random matrix with std `scale`. Returns [m, newSeed].
fn randn(r, c, seed, scale) {
    int i = 0; int j = 0; int m = 0; int row = 0; int rr = 0;
    if (scale == nil) { scale = 1.0; }
    m = [];
    for (i = 0; i < r; ++i) {
        row = [];
        for (j = 0; j < c; ++j) {
            rr = prng.normal(seed);
            row[j] = rr[0] * scale;
            seed = rr[1];
        }
        m[i] = row;
    }
    return [m, seed];
}

fn add(a, b) {
    int c = 0; int i = 0; int j = 0; int out = 0; int r = 0; int ra = 0; int rb = 0; int row = 0;
    r = rows(a); c = cols(a);
    out = [];
    for (i = 0; i < r; ++i) {
        ra = a[i]; rb = b[i];
        row = [];
        for (j = 0; j < c; ++j) { row[j] = ra[j] + rb[j]; }
        out[i] = row;
    }
    return out;
}

fn sub(a, b) {
    int c = 0; int i = 0; int j = 0; int out = 0; int r = 0; int ra = 0; int rb = 0; int row = 0;
    r = rows(a); c = cols(a);
    out = [];
    for (i = 0; i < r; ++i) {
        ra = a[i]; rb = b[i];
        row = [];
        for (j = 0; j < c; ++j) { row[j] = ra[j] - rb[j]; }
        out[i] = row;
    }
    return out;
}

fn hadamard(a, b) {
    int c = 0; int i = 0; int j = 0; int out = 0; int r = 0; int ra = 0; int rb = 0; int row = 0;
    r = rows(a); c = cols(a);
    out = [];
    for (i = 0; i < r; ++i) {
        ra = a[i]; rb = b[i];
        row = [];
        for (j = 0; j < c; ++j) { row[j] = ra[j] * rb[j]; }
        out[i] = row;
    }
    return out;
}

fn scale(a, s) {
    int c = 0; int i = 0; int j = 0; int out = 0; int r = 0; int ra = 0; int row = 0;
    r = rows(a); c = cols(a);
    out = [];
    for (i = 0; i < r; ++i) {
        ra = a[i];
        row = [];
        for (j = 0; j < c; ++j) { row[j] = ra[j] * s; }
        out[i] = row;
    }
    return out;
}

fn add_scalar(a, s) {
    int c = 0; int i = 0; int j = 0; int out = 0; int r = 0; int ra = 0; int row = 0;
    r = rows(a); c = cols(a);
    out = [];
    for (i = 0; i < r; ++i) {
        ra = a[i];
        row = [];
        for (j = 0; j < c; ++j) { row[j] = ra[j] + s; }
        out[i] = row;
    }
    return out;
}

# Apply a one-argument function to every element.
fn apply(a, f) {
    int c = 0; int i = 0; int j = 0; int out = 0; int r = 0; int ra = 0; int row = 0;
    r = rows(a); c = cols(a);
    out = [];
    for (i = 0; i < r; ++i) {
        ra = a[i];
        row = [];
        for (j = 0; j < c; ++j) { row[j] = f(ra[j]); }
        out[i] = row;
    }
    return out;
}

fn matmul(a, b) {
    int aik = 0; int i = 0; int j = 0; int k = 0; int m = 0; int n = 0; int out = 0; int p = 0; int ra = 0; int rb = 0; int row = 0;
    n = rows(a); m = cols(a); p = cols(b);
    out = [];
    for (i = 0; i < n; ++i) {
        ra = a[i];
        row = [];
        for (j = 0; j < p; ++j) { row[j] = 0.0; }
        for (k = 0; k < m; ++k) {
            aik = ra[k];
            if (aik != 0.0) {
                rb = b[k];
                for (j = 0; j < p; ++j) { row[j] = row[j] + aik * rb[j]; }
            }
        }
        out[i] = row;
    }
    return out;
}

# Matrix times column vector -> vector.
fn matvec(a, v) {
    int i = 0; int k = 0; int m = 0; int n = 0; int out = 0; int ra = 0; int s = 0;
    n = rows(a); m = cols(a);
    out = [];
    for (i = 0; i < n; ++i) {
        ra = a[i];
        s = 0.0;
        for (k = 0; k < m; ++k) { s = s + ra[k] * v[k]; }
        out[i] = s;
    }
    return out;
}

# Row vector times matrix -> vector.
fn vecmat(v, a) {
    int j = 0; int k = 0; int m = 0; int out = 0; int p = 0; int ra = 0; int vk = 0;
    m = rows(a); p = cols(a);
    out = [];
    for (j = 0; j < p; ++j) { out[j] = 0.0; }
    for (k = 0; k < m; ++k) {
        vk = v[k];
        ra = a[k];
        for (j = 0; j < p; ++j) { out[j] = out[j] + vk * ra[j]; }
    }
    return out;
}

fn outer(u, v) {
    int i = 0; int j = 0; int out = 0; int row = 0;
    out = [];
    for (i = 0; i < len(u); ++i) {
        row = [];
        for (j = 0; j < len(v); ++j) { row[j] = u[i] * v[j]; }
        out[i] = row;
    }
    return out;
}

fn transpose(a) {
    int c = 0; int i = 0; int j = 0; int out = 0; int r = 0; int row = 0;
    r = rows(a); c = cols(a);
    out = [];
    for (j = 0; j < c; ++j) {
        row = [];
        for (i = 0; i < r; ++i) { row[i] = a[i][j]; }
        out[j] = row;
    }
    return out;
}

fn trace(a) {
    int i = 0; int n = 0; int s = 0;
    s = 0.0;
    n = mathx.minv(rows(a), cols(a));
    for (i = 0; i < n; ++i) { s = s + a[i][i]; }
    return s;
}

fn row(m, i) { return m[i]; }

fn col(m, j) {
    int i = 0; int out = 0;
    out = [];
    for (i = 0; i < rows(m); ++i) { out[i] = m[i][j]; }
    return out;
}

fn set_row(m, i, v) { m[i] = v; return m; }

fn set_col(m, j, v) {
    int i = 0;
    for (i = 0; i < rows(m); ++i) { m[i][j] = v[i]; }
    return m;
}

fn swap_rows(m, i, j) {
    int t = 0;
    t = m[i]; m[i] = m[j]; m[j] = t;
    return m;
}

fn hcat(a, b) {
    int i = 0; int out = 0;
    out = [];
    for (i = 0; i < rows(a); ++i) { out[i] = a[i] + b[i]; }
    return out;
}

fn vcat(a, b) {
    int i = 0; int out = 0;
    out = [];
    for (i = 0; i < rows(a); ++i) { out[len(out)] = a[i]; }
    for (i = 0; i < rows(b); ++i) { out[len(out)] = b[i]; }
    return out;
}

# Rows [r0, r1) and columns [c0, c1).
fn submatrix(m, r0, r1, c0, c1) {
    int i = 0; int j = 0; int out = 0; int row = 0; int src = 0;
    out = [];
    for (i = r0; i < r1; ++i) {
        row = [];
        src = m[i];
        for (j = c0; j < c1; ++j) { row[len(row)] = src[j]; }
        out[len(out)] = row;
    }
    return out;
}

fn norm_fro(a) {
    int i = 0; int j = 0; int ra = 0; int s = 0;
    s = 0.0;
    for (i = 0; i < rows(a); ++i) {
        ra = a[i];
        for (j = 0; j < len(ra); ++j) { s = s + ra[j] * ra[j]; }
    }
    return sqrt(s);
}

fn norm_max(a) {
    int i = 0; int j = 0; int ra = 0; int s = 0; int v = 0;
    s = 0.0;
    for (i = 0; i < rows(a); ++i) {
        ra = a[i];
        for (j = 0; j < len(ra); ++j) {
            v = abs(ra[j]);
            if (v > s) { s = v; }
        }
    }
    return s;
}

fn col_means(a) {
    int c = 0; int i = 0; int j = 0; int out = 0; int r = 0; int ra = 0;
    r = rows(a); c = cols(a);
    out = [];
    for (j = 0; j < c; ++j) { out[j] = 0.0; }
    for (i = 0; i < r; ++i) {
        ra = a[i];
        for (j = 0; j < c; ++j) { out[j] = out[j] + ra[j]; }
    }
    for (j = 0; j < c; ++j) { out[j] = out[j] / r; }
    return out;
}

fn print_mat(m, decimals) {
    int i = 0; int j = 0; int line = 0; int ra = 0;
    if (decimals == nil) { decimals = 4; }
    for (i = 0; i < rows(m); ++i) {
        line = "[ ";
        ra = m[i];
        for (j = 0; j < len(ra); ++j) {
            line = line + strx.pad_left(strx.fmt_f(ra[j], decimals), decimals + 6, " ") + " ";
        }
        print(line + "]");
    }
}

# ---------------------------------------------------------------------------
# LU decomposition with partial pivoting.
# Returns [LU, perm, parity, ok]. LU stores L (unit diagonal, below) and U
# (on and above the diagonal). perm[i] is the source row of pivoted row i.
# ---------------------------------------------------------------------------
fn lu(aIn) {
    int a = 0; int best = 0; int f = 0; int i = 0; int j = 0; int k = 0; int n = 0; int ok = 0; int p = 0; int parity = 0; int perm = 0; int piv = 0; int ri = 0; int rk = 0; int t = 0; int v = 0;
    n = rows(aIn);
    a = aIn;
    perm = [];
    for (i = 0; i < n; ++i) { perm[i] = i; }
    parity = 1;
    ok = true;
    for (k = 0; k < n; ++k) {
        # pivot search
        p = k;
        best = abs(a[k][k]);
        for (i = k + 1; i < n; ++i) {
            v = abs(a[i][k]);
            if (v > best) { best = v; p = i; }
        }
        if (best < 1.0e-13) { ok = false; continue; }
        if (p != k) {
            a = swap_rows(a, p, k);
            t = perm[p]; perm[p] = perm[k]; perm[k] = t;
            parity = 0 - parity;
        }
        piv = a[k][k];
        for (i = k + 1; i < n; ++i) {
            f = a[i][k] / piv;
            a[i][k] = f;
            if (f != 0.0) {
                ri = a[i];
                rk = a[k];
                for (j = k + 1; j < n; ++j) { ri[j] = ri[j] - f * rk[j]; }
                a[i] = ri;
            }
        }
    }
    return [a, perm, parity, ok];
}

fn det(aIn) {
    int d = 0; int i = 0; int m = 0; int r = 0;
    r = lu(aIn);
    if (!r[3]) { return 0.0; }
    m = r[0];
    d = r[2] * 1.0;
    for (i = 0; i < rows(m); ++i) { d = d * m[i][i]; }
    return d;
}

# Solve A x = b for a vector b using a precomputed lu() result.
fn lu_solve(luRes, b) {
    int d = 0; int i = 0; int j = 0; int m = 0; int n = 0; int perm = 0; int ri = 0; int s = 0; int x = 0; int y = 0;
    m = luRes[0];
    perm = luRes[1];
    n = rows(m);
    # forward substitution with permutation
    y = [];
    for (i = 0; i < n; ++i) {
        s = b[perm[i]];
        ri = m[i];
        for (j = 0; j < i; ++j) { s = s - ri[j] * y[j]; }
        y[i] = s;
    }
    # back substitution
    x = [];
    for (i = 0; i < n; ++i) { x[i] = 0.0; }
    for (i = n - 1; i >= 0; --i) {
        s = y[i];
        ri = m[i];
        for (j = i + 1; j < n; ++j) { s = s - ri[j] * x[j]; }
        d = ri[i];
        if (abs(d) < 1.0e-13) { x[i] = 0.0; }
        else { x[i] = s / d; }
    }
    return x;
}

# Solve A x = b. Returns the solution vector, or nil if A is singular.
fn solve(a, b) {
    int r = 0;
    r = lu(a);
    if (!r[3]) { return nil; }
    return lu_solve(r, b);
}

fn inverse(a) {
    int ej = 0; int i = 0; int j = 0; int n = 0; int outT = 0; int r = 0;
    n = rows(a);
    r = lu(a);
    if (!r[3]) { return nil; }
    outT = [];
    for (j = 0; j < n; ++j) {
        ej = [];
        for (i = 0; i < n; ++i) {
            if (i == j) { ej[i] = 1.0; } else { ej[i] = 0.0; }
        }
        outT[j] = lu_solve(r, ej);   # column j of the inverse, as a row here
    }
    return transpose(outT);
}

# ---------------------------------------------------------------------------
# QR via modified Gram-Schmidt. a is r x c with r >= c. Returns [Q, R].
# ---------------------------------------------------------------------------
fn qr(a) {
    int c = 0; int i = 0; int j = 0; int k = 0; int nn = 0; int q = 0; int qi = 0; int r = 0; int rm = 0; int s = 0; int v = 0; int vj = 0;
    r = rows(a); c = cols(a);
    v = transpose(a);          # work on columns as rows
    q = [];
    rm = zeros(c, c);
    for (j = 0; j < c; ++j) {
        vj = v[j];
        for (i = 0; i < j; ++i) {
            qi = q[i];
            s = 0.0;
            for (k = 0; k < r; ++k) { s = s + qi[k] * vj[k]; }
            rm[i][j] = s;
            for (k = 0; k < r; ++k) { vj[k] = vj[k] - s * qi[k]; }
        }
        nn = 0.0;
        for (k = 0; k < r; ++k) { nn = nn + vj[k] * vj[k]; }
        nn = sqrt(nn);
        rm[j][j] = nn;
        if (nn < 1.0e-13) {
            for (k = 0; k < r; ++k) { vj[k] = 0.0; }
        } else {
            for (k = 0; k < r; ++k) { vj[k] = vj[k] / nn; }
        }
        q[j] = vj;
    }
    return [transpose(q), rm];
}

# Least squares solution of A x ~= b via QR (A is r x c, r >= c).
fn lstsq(a, b) {
    int c = 0; int d = 0; int i = 0; int j = 0; int q = 0; int qrres = 0; int qtb = 0; int ri = 0; int rm = 0; int s = 0; int x = 0;
    qrres = qr(a);
    q = qrres[0];
    rm = qrres[1];
    c = cols(a);
    qtb = matvec(transpose(q), b);
    x = [];
    for (i = 0; i < c; ++i) { x[i] = 0.0; }
    for (i = c - 1; i >= 0; --i) {
        s = qtb[i];
        ri = rm[i];
        for (j = i + 1; j < c; ++j) { s = s - ri[j] * x[j]; }
        d = ri[i];
        if (abs(d) < 1.0e-13) { x[i] = 0.0; }
        else { x[i] = s / d; }
    }
    return x;
}

# ---------------------------------------------------------------------------
# Power iteration: dominant eigenpair. Returns [lambda, vec, newSeed].
# ---------------------------------------------------------------------------
fn eig_power(a, iters, seed) {
    int i = 0; int lam = 0; int n = 0; int nn = 0; int rr = 0; int t = 0; int v = 0; int w = 0; int wv = 0;
    if (iters == nil) { iters = 100; }
    n = rows(a);
    rr = prng.randf_array(seed, n, -1.0, 1.0);
    v = rr[0];
    seed = rr[1];
    lam = 0.0;
    for (t = 0; t < iters; ++t) {
        w = matvec(a, v);
        nn = 0.0;
        for (i = 0; i < n; ++i) { nn = nn + w[i] * w[i]; }
        nn = sqrt(nn);
        if (nn < 1.0e-14) { break; }
        for (i = 0; i < n; ++i) { v[i] = w[i] / nn; }
        wv = matvec(a, v);
        lam = 0.0;
        for (i = 0; i < n; ++i) { lam = lam + v[i] * wv[i]; }
    }
    return [lam, v, seed];
}

# ---------------------------------------------------------------------------
# Cyclic Jacobi eigendecomposition for SYMMETRIC matrices.
# Returns [eigenvalues, V] with eigenvalues sorted descending and the columns
# of V the matching orthonormal eigenvectors (A = V diag(vals) V^T).
# ---------------------------------------------------------------------------
fn eig_jacobi(aIn, maxSweeps) {
    int a = 0; int akp = 0; int akq = 0; int ap = 0; int apk = 0; int apq = 0; int aqk = 0; int bi = 0; int cth = 0; int i = 0; int idx = 0; int j = 0; int k = 0; int kk = 0; int n = 0; int off = 0; int p = 0; int q = 0; int sortedVals = 0; int src = 0; int sth = 0; int sweep = 0; int t = 0; int tau = 0; int v = 0; int vals = 0; int vkp = 0; int vkq = 0; int vv = 0;
    if (maxSweeps == nil) { maxSweeps = 60; }
    n = rows(aIn);
    a = aIn;
    v = identity(n);
    for (sweep = 0; sweep < maxSweeps; ++sweep) {
        off = 0.0;
        for (p = 0; p < n; ++p) {
            ap = a[p];
            for (q = p + 1; q < n; ++q) { off = off + ap[q] * ap[q]; }
        }
        if (off < 1.0e-22) { break; }
        for (p = 0; p < n - 1; ++p) {
            for (q = p + 1; q < n; ++q) {
                apq = a[p][q];
                if (abs(apq) < 1.0e-15) { continue; }
                tau = (a[q][q] - a[p][p]) / (2.0 * apq);
                t = 0.0;
                if (tau >= 0) { t = 1.0 / (tau + sqrt(1.0 + tau * tau)); }
                else { t = -1.0 / ((0.0 - tau) + sqrt(1.0 + tau * tau)); }
                cth = 1.0 / sqrt(1.0 + t * t);
                sth = t * cth;
                # rotate rows/cols p and q of a
                for (k = 0; k < n; ++k) {
                    akp = a[k][p];
                    akq = a[k][q];
                    a[k][p] = cth * akp - sth * akq;
                    a[k][q] = sth * akp + cth * akq;
                }
                for (k = 0; k < n; ++k) {
                    apk = a[p][k];
                    aqk = a[q][k];
                    a[p][k] = cth * apk - sth * aqk;
                    a[q][k] = sth * apk + cth * aqk;
                }
                for (k = 0; k < n; ++k) {
                    vkp = v[k][p];
                    vkq = v[k][q];
                    v[k][p] = cth * vkp - sth * vkq;
                    v[k][q] = sth * vkp + cth * vkq;
                }
            }
        }
    }
    vals = get_diag(a);
    # sort descending, reorder eigenvector columns to match
    idx = [];
    for (i = 0; i < n; ++i) { idx[i] = i; }
    for (i = 0; i < n; ++i) {
        bi = i;
        for (j = i + 1; j < n; ++j) { if (vals[idx[j]] > vals[idx[bi]]) { bi = j; } }
        t = idx[i]; idx[i] = idx[bi]; idx[bi] = t;
    }
    sortedVals = [];
    vv = zeros(n, n);
    for (kk = 0; kk < n; ++kk) {
        src = idx[kk];
        sortedVals[kk] = vals[src];
        for (i = 0; i < n; ++i) { vv[i][kk] = v[i][src]; }
    }
    return [sortedVals, vv];
}

# ---------------------------------------------------------------------------
# SVD of an r x c matrix (r >= 1, c >= 1) via eigen of A^T A.
# Returns [U, svals, V]: A = U diag(svals) V^T, svals descending.
# Columns of U beyond the numerical rank are zero.
# ---------------------------------------------------------------------------
fn svd(a) {
    int at = 0; int ata = 0; int avj = 0; int c = 0; int eig = 0; int i = 0; int j = 0; int lams = 0; int lv = 0; int r = 0; int svals = 0; int u = 0; int v = 0; int vj = 0;
    int cand = 0; int dt = 0; int nrm2 = 0; int jj = 0; int k = 0;
    int smax2 = 0;
    int stol = 0;
    r = rows(a); c = cols(a);
    at = transpose(a);
    ata = matmul(at, a);
    eig = eig_jacobi(ata, nil);
    lams = eig[0];
    v = eig[1];
    svals = [];
    for (i = 0; i < c; ++i) {
        lv = lams[i];
        if (lv < 0.0) { lv = 0.0; }
        svals[i] = sqrt(lv);
    }
    # rank cutoff RELATIVE to the largest singular value: tiny-but-nonzero
    # values are eigensolver roundoff, and dividing by them would fill U
    # with noise instead of a usable left-null-space column.
    smax2 = 0.0;
    for (j = 0; j < c; ++j) { if (svals[j] > smax2) { smax2 = svals[j]; } }
    stol = smax2 * 1.0e-8 + 1.0e-12;
    u = zeros(r, c);
    for (j = 0; j < c; ++j) {
        if (svals[j] > stol) {
            vj = col(v, j);
            avj = matvec(a, vj);
            for (i = 0; i < r; ++i) { u[i][j] = avj[i] / svals[j]; }
        }
    }
    # complete rank-deficient columns of U to an orthonormal set (needed by
    # consumers that use the left null space, e.g. essential-matrix pose
    # recovery). Gram-Schmidt of basis vectors against the existing columns.
    for (j = 0; j < c; ++j) {
        if (svals[j] > stol) { continue; }
        for (k = 0; k < r; ++k) {
            cand = [];
            for (i = 0; i < r; ++i) { cand[i] = 0.0; }
            cand[k] = 1.0;
            for (jj = 0; jj < c; ++jj) {
                if (jj == j) { continue; }
                dt = 0.0;
                for (i = 0; i < r; ++i) { dt = dt + u[i][jj] * cand[i]; }
                for (i = 0; i < r; ++i) { cand[i] = cand[i] - dt * u[i][jj]; }
            }
            nrm2 = 0.0;
            for (i = 0; i < r; ++i) { nrm2 = nrm2 + cand[i] * cand[i]; }
            if (nrm2 > 1.0e-8) {
                nrm2 = sqrt(nrm2);
                for (i = 0; i < r; ++i) { u[i][j] = cand[i] / nrm2; }
                break;
            }
        }
    }
    return [u, svals, v];
}

# Right-singular vector for the SMALLEST singular value: the least-squares
# unit solution of A x = 0 (the workhorse behind DLT / 8-point algorithms).
fn null_vector(a) {
    int s = 0; int v = 0;
    s = svd(a);
    v = s[2];
    return col(v, cols(a) - 1);
}

# Moore-Penrose pseudoinverse via SVD.
fn pinv(a, tol) {
    int c = 0; int i = 0; int s = 0; int sinv = 0; int smax = 0; int sv = 0; int u = 0; int v = 0;
    s = svd(a);
    u = s[0];
    sv = s[1];
    v = s[2];
    c = len(sv);
    if (tol == nil) {
        smax = 0.0;
        for x in sv { if (x > smax) { smax = x; } }
        tol = 1.0e-10 * mathx.maxv(1.0, smax);
    }
    sinv = [];
    for (i = 0; i < c; ++i) {
        if (sv[i] > tol) { sinv[i] = 1.0 / sv[i]; }
        else { sinv[i] = 0.0; }
    }
    return matmul(matmul(v, diag(sinv)), transpose(u));
}

fn rank(a, tol) {
    int r = 0; int s = 0; int smax = 0; int sv = 0;
    s = svd(a);
    sv = s[1];
    smax = 0.0;
    for x in sv { if (x > smax) { smax = x; } }
    if (tol == nil) { tol = 1.0e-9 * mathx.maxv(1.0, smax); }
    r = 0;
    for x in sv { if (x > tol) { r = r + 1; } }
    return r;
}
