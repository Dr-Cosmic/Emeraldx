# =============================================================================
# dataset.em - loading, splitting and preparing tabular data
#
# A dataset is [X, y]: X an array of feature-row arrays, y an array of
# targets (numbers or strings before encoding).
# =============================================================================

import strx;
import arrayx;
import stats;
import prng;
import dictx;
# emx-scope-safe

# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

# Read a whole file as one string ("" if unreadable).
fn read_text(path) {
    int s = 0;
    File fh = path;
    s = fh.read();
    if (s == nil) { return ""; }
    return s;
}

fn write_text(path, contents) {
    File fh = path;
    fh.write(contents);
    return true;
}

# Parse simple CSV text (no quoted commas) into an array of string rows.
fn parse_csv(text) {
    int lines = 0; int rows = 0; int t = 0;
    rows = [];
    lines = strx.split_lines(text);
    for ln in lines {
        t = strx.trim(ln);
        if (len(t) == 0) { continue; }
        rows[len(rows)] = strx.split(t, ",");
    }
    return rows;
}

# Load a numeric CSV: returns [X, y, header].
# If hasHeader, the first row becomes `header`. The column `targetCol`
# (default: last) becomes y; the rest become X. Non-numeric cells parse to nil
# and are replaced by 0.0 in X; y keeps strings (for classification labels).
fn load_csv(path, hasHeader, targetCol) {
    int feats = 0; int header = 0; int i = 0; int jj = 0; int ncol = 0; int r = 0; int rows = 0; int start = 0; int tc = 0; int text = 0; int v = 0; int xs = 0; int ys = 0; int yv = 0;
    text = read_text(path);
    rows = parse_csv(text);
    header = [];
    start = 0;
    if (hasHeader == true && len(rows) > 0) {
        header = rows[0];
        start = 1;
    }
    xs = [];
    ys = [];
    for (i = start; i < len(rows); ++i) {
        r = rows[i];
        ncol = len(r);
        tc = targetCol;
        if (tc == nil) { tc = ncol - 1; }
        feats = [];
        for (jj = 0; jj < ncol; ++jj) {
            if (jj == tc) { continue; }
            v = strx.parse_float(r[jj]);
            if (v == nil) { v = 0.0; }
            feats[len(feats)] = v;
        }
        yv = strx.parse_float(r[tc]);
        if (yv == nil) { yv = r[tc]; }
        xs[len(xs)] = feats;
        ys[len(ys)] = yv;
    }
    return [xs, ys, header];
}

fn save_csv(path, xrows, yvals, header) {
    int i = 0; int line = 0; int text = 0;
    text = "";
    if (header != nil && len(header) > 0) {
        text = strx.join(header, ",") + "\n";
    }
    for (i = 0; i < len(xrows); ++i) {
        line = strx.join(xrows[i], ",");
        if (yvals != nil) { line = line + "," + string(yvals[i]); }
        text = text + line + "\n";
    }
    File fh = path;
    fh.write(text);
    return true;
}

# ---------------------------------------------------------------------------
# Splitting / shuffling
# ---------------------------------------------------------------------------

# Shuffle rows of X and y together. Returns [X, y, newSeed].
fn shuffle_xy(xs, ys, seed) {
    int i = 0; int idx = 0; int n = 0; int nx = 0; int ny = 0; int perm = 0; int r = 0;
    n = len(xs);
    idx = [];
    for (i = 0; i < n; ++i) { idx[i] = i; }
    r = arrayx.shuffle(idx, seed);
    perm = r[0];
    nx = [];
    ny = [];
    for (i = 0; i < n; ++i) {
        nx[i] = xs[perm[i]];
        ny[i] = ys[perm[i]];
    }
    return [nx, ny, r[1]];
}

# Split into train/test. Returns [xTrain, yTrain, xTest, yTest, newSeed].
fn train_test_split(xs, ys, testFrac, seed) {
    int i = 0; int n = 0; int nTest = 0; int r = 0; int sx = 0; int sy = 0; int xTe = 0; int xTr = 0; int yTe = 0; int yTr = 0;
    r = shuffle_xy(xs, ys, seed);
    sx = r[0];
    sy = r[1];
    n = len(sx);
    nTest = int(floor(n * testFrac));
    if (nTest < 1) { nTest = 1; }
    xTr = []; yTr = []; xTe = []; yTe = [];
    for (i = 0; i < n; ++i) {
        if (i < n - nTest) {
            xTr[len(xTr)] = sx[i];
            yTr[len(yTr)] = sy[i];
        } else {
            xTe[len(xTe)] = sx[i];
            yTe[len(yTe)] = sy[i];
        }
    }
    return [xTr, yTr, xTe, yTe, r[2]];
}

# K-fold index sets: returns an array of k arrays of row indices.
fn kfold_indices(n, k, seed) {
    int cur = 0; int f = 0; int folds = 0; int i = 0; int idx = 0; int perm = 0; int r = 0;
    idx = [];
    for (i = 0; i < n; ++i) { idx[i] = i; }
    r = arrayx.shuffle(idx, seed);
    perm = r[0];
    folds = [];
    for (f = 0; f < k; ++f) { folds[f] = []; }
    for (i = 0; i < n; ++i) {
        f = i % k;
        cur = folds[f];
        cur[len(cur)] = perm[i];
        folds[f] = cur;
    }
    return folds;
}

# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

# Fit column means/stds on X. Returns [means, stds].
fn fit_standardizer(xs) {
    int d = 0; int dv = 0; int i = 0; int j = 0; int means = 0; int n = 0; int row = 0; int stds = 0;
    n = len(xs);
    d = len(xs[0]);
    means = arrayx.zerosf(d);
    for (i = 0; i < n; ++i) {
        row = xs[i];
        for (j = 0; j < d; ++j) { means[j] = means[j] + row[j]; }
    }
    for (j = 0; j < d; ++j) { means[j] = means[j] / n; }
    stds = arrayx.zerosf(d);
    for (i = 0; i < n; ++i) {
        row = xs[i];
        for (j = 0; j < d; ++j) {
            dv = row[j] - means[j];
            stds[j] = stds[j] + dv * dv;
        }
    }
    for (j = 0; j < d; ++j) {
        stds[j] = sqrt(stds[j] / n);
        if (stds[j] < 1.0e-12) { stds[j] = 1.0; }
    }
    return [means, stds];
}

fn apply_standardizer(xs, stdzr) {
    int i = 0; int j = 0; int means = 0; int nr = 0; int out = 0; int row = 0; int stds = 0;
    means = stdzr[0];
    stds = stdzr[1];
    out = [];
    for (i = 0; i < len(xs); ++i) {
        row = xs[i];
        nr = [];
        for (j = 0; j < len(row); ++j) { nr[j] = (row[j] - means[j]) / stds[j]; }
        out[i] = nr;
    }
    return out;
}

# Map arbitrary labels to 0..k-1. Returns [encodedY, classes].
fn encode_labels(ys) {
    int c = 0; int classes = 0; int enc = 0; int found = 0; int i = 0;
    classes = [];
    enc = [];
    for (i = 0; i < len(ys); ++i) {
        found = -1;
        for (c = 0; c < len(classes); ++c) {
            if (classes[c] == ys[i]) { found = c; break; }
        }
        if (found < 0) {
            found = len(classes);
            classes[len(classes)] = ys[i];
        }
        enc[i] = found;
    }
    return [enc, classes];
}

# One-hot encode integer labels 0..k-1 into rows of length k.
fn one_hot(ys, k) {
    int i = 0; int out = 0; int row = 0;
    out = [];
    for (i = 0; i < len(ys); ++i) {
        row = arrayx.zerosf(k);
        row[ys[i]] = 1.0;
        out[i] = row;
    }
    return out;
}

# ---------------------------------------------------------------------------
# Toy data generators (each returns [X, y, newSeed])
# ---------------------------------------------------------------------------

# Two gaussian blobs around (-d, -d) and (+d, +d), labels 0/1.
fn make_blobs(nPerClass, dist, spread, seed) {
    int c = 0; int cx = 0; int i = 0; int r1 = 0; int r2 = 0; int xs = 0; int ys = 0;
    if (spread == nil) { spread = 1.0; }
    xs = [];
    ys = [];
    for (c = 0; c < 2; ++c) {
        cx = dist;
        if (c == 0) { cx = 0.0 - dist; }
        for (i = 0; i < nPerClass; ++i) {
            r1 = prng.normal(seed);
            r2 = prng.normal(r1[1]);
            seed = r2[1];
            xs[len(xs)] = [cx + r1[0] * spread, cx + r2[0] * spread];
            ys[len(ys)] = c;
        }
    }
    return [xs, ys, seed];
}

# Two interleaved half-moons, labels 0/1.
fn make_moons(nPerClass, noise, seed) {
    int i = 0; int r1 = 0; int r2 = 0; int r3 = 0; int r4 = 0; int t = 0; int xs = 0; int ys = 0;
    if (noise == nil) { noise = 0.1; }
    xs = [];
    ys = [];
    for (i = 0; i < nPerClass; ++i) {
        t = pi * i / (nPerClass - 1.0);
        r1 = prng.normal(seed);
        r2 = prng.normal(r1[1]);
        seed = r2[1];
        xs[len(xs)] = [cos(t) + r1[0] * noise, sin(t) + r2[0] * noise];
        ys[len(ys)] = 0;
        r3 = prng.normal(seed);
        r4 = prng.normal(r3[1]);
        seed = r4[1];
        xs[len(xs)] = [1.0 - cos(t) + r3[0] * noise, 0.5 - sin(t) + r4[0] * noise];
        ys[len(ys)] = 1;
    }
    return [xs, ys, seed];
}

# y = w.x + b + gaussian noise. Returns [X, y, trueW, seed].
fn make_regression(n, w, b, noise, seed) {
    int d = 0; int i = 0; int j = 0; int r = 0; int rn = 0; int row = 0; int s = 0; int xs = 0; int ys = 0;
    d = len(w);
    xs = [];
    ys = [];
    for (i = 0; i < n; ++i) {
        row = [];
        s = b;
        for (j = 0; j < d; ++j) {
            r = prng.uniform(seed, -2.0, 2.0);
            row[j] = r[0];
            seed = r[1];
            s = s + w[j] * row[j];
        }
        rn = prng.normal(seed);
        seed = rn[1];
        xs[i] = row;
        ys[i] = s + rn[0] * noise;
    }
    return [xs, ys, w, seed];
}

# XOR-ish 2D classification data in [-1,1]^2.
fn make_xor(n, seed) {
    int i = 0; int lbl = 0; int r1 = 0; int r2 = 0; int x0 = 0; int x1 = 0; int xs = 0; int ys = 0;
    xs = [];
    ys = [];
    for (i = 0; i < n; ++i) {
        r1 = prng.uniform(seed, -1.0, 1.0);
        r2 = prng.uniform(r1[1], -1.0, 1.0);
        seed = r2[1];
        x0 = r1[0];
        x1 = r2[0];
        lbl = 0;
        if ((x0 > 0.0 && x1 <= 0.0) || (x0 <= 0.0 && x1 > 0.0)) { lbl = 1; }
        xs[len(xs)] = [x0, x1];
        ys[len(ys)] = lbl;
    }
    return [xs, ys, seed];
}
