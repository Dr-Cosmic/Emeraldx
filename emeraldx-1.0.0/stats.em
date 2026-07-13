# =============================================================================
# stats.em - descriptive statistics
# =============================================================================

import mathx;
import arrayx;
# emx-scope-safe

fn mean(xs) {
    int s = 0;
    if (len(xs) == 0) { return 0.0; }
    s = 0.0;
    for x in xs { s = s + x; }
    return s / len(xs);
}

# Population variance if ddof = 0 (default), sample variance if ddof = 1.
fn variance(xs, ddof) {
    int m = 0; int n = 0; int s = 0;
    if (ddof == nil) { ddof = 0; }
    n = len(xs);
    if (n - ddof <= 0) { return 0.0; }
    m = mean(xs);
    s = 0.0;
    for x in xs { s = s + (x - m) * (x - m); }
    return s / (n - ddof);
}

fn stddev(xs, ddof) { return sqrt(variance(xs, ddof)); }

fn median(xs) {
    int h = 0; int n = 0; int s = 0;
    n = len(xs);
    if (n == 0) { return 0.0; }
    s = arrayx.sort(xs);
    h = n // 2;
    if (n % 2 == 1) { return s[h]; }
    return (s[h - 1] + s[h]) / 2.0;
}

# Linear-interpolated quantile, q in [0, 1].
fn quantile(xs, q) {
    int frac = 0; int hi = 0; int lo = 0; int n = 0; int pos = 0; int s = 0;
    n = len(xs);
    if (n == 0) { return 0.0; }
    s = arrayx.sort(xs);
    pos = q * (n - 1);
    lo = int(floor(pos));
    hi = lo + 1;
    if (hi >= n) { return s[n - 1]; }
    frac = pos - lo;
    return s[lo] * (1.0 - frac) + s[hi] * frac;
}

fn mode_of(xs) {
    int best = 0; int bestCount = 0; int count = 0; int cur = 0; int i = 0; int n = 0; int s = 0;
    n = len(xs);
    if (n == 0) { return 0.0; }
    s = arrayx.sort(xs);
    best = s[0];
    bestCount = 1;
    cur = s[0];
    count = 1;
    for (i = 1; i < n; ++i) {
        if (s[i] == cur) { count = count + 1; }
        else { cur = s[i]; count = 1; }
        if (count > bestCount) { bestCount = count; best = cur; }
    }
    return best;
}

fn covariance(xs, ys, ddof) {
    int i = 0; int mx = 0; int my = 0; int n = 0; int s = 0;
    if (ddof == nil) { ddof = 0; }
    n = len(xs);
    if (n - ddof <= 0) { return 0.0; }
    mx = mean(xs);
    my = mean(ys);
    s = 0.0;
    for (i = 0; i < n; ++i) { s = s + (xs[i] - mx) * (ys[i] - my); }
    return s / (n - ddof);
}

# Pearson correlation coefficient.
fn correlation(xs, ys) {
    int sx = 0; int sy = 0;
    sx = stddev(xs, 0);
    sy = stddev(ys, 0);
    if (sx < 1.0e-12 || sy < 1.0e-12) { return 0.0; }
    return covariance(xs, ys, 0) / (sx * sy);
}

# Spearman rank correlation.
fn spearman(xs, ys) {
    int rx = 0; int ry = 0;
    rx = ranks(xs);
    ry = ranks(ys);
    return correlation(rx, ry);
}

# Average ranks (ties get the mean of their positions), 1-based.
fn ranks(xs) {
    int avg = 0; int i = 0; int idx = 0; int j = 0; int k = 0; int n = 0; int r = 0;
    n = len(xs);
    idx = arrayx.argsort(xs);
    r = arrayx.zerosf(n);
    i = 0;
    while (i < n) {
        j = i;
        while (j + 1 < n && xs[idx[j + 1]] == xs[idx[i]]) { j = j + 1; }
        avg = (i + j) / 2.0 + 1.0;
        for (k = i; k <= j; ++k) { r[idx[k]] = avg; }
        i = j + 1;
    }
    return r;
}

fn skewness(xs) {
    int m = 0; int n = 0; int s = 0; int sd = 0;
    n = len(xs);
    if (n < 2) { return 0.0; }
    m = mean(xs);
    sd = stddev(xs, 0);
    if (sd < 1.0e-12) { return 0.0; }
    s = 0.0;
    for x in xs { s = s + ((x - m) / sd) ** 3; }
    return s / n;
}

fn kurtosis(xs) {
    int m = 0; int n = 0; int s = 0; int sd = 0;
    n = len(xs);
    if (n < 2) { return 0.0; }
    m = mean(xs);
    sd = stddev(xs, 0);
    if (sd < 1.0e-12) { return 0.0; }
    s = 0.0;
    for x in xs { s = s + ((x - m) / sd) ** 4; }
    return s / n - 3.0;
}

# [zscores], using population std.
fn zscores(xs) {
    int i = 0; int m = 0; int out = 0; int sd = 0;
    m = mean(xs);
    sd = stddev(xs, 0);
    out = [];
    if (sd < 1.0e-12) { sd = 1.0; }
    for (i = 0; i < len(xs); ++i) { out[i] = (xs[i] - m) / sd; }
    return out;
}

# Scale to [0, 1]. Constant input maps to 0.5.
fn minmax_scale(xs) {
    int hi = 0; int i = 0; int lo = 0; int out = 0;
    lo = arrayx.minv(xs);
    hi = arrayx.maxv(xs);
    out = [];
    if (hi - lo < 1.0e-12) {
        for (i = 0; i < len(xs); ++i) { out[i] = 0.5; }
        return out;
    }
    for (i = 0; i < len(xs); ++i) { out[i] = (xs[i] - lo) / (hi - lo); }
    return out;
}

# Histogram over [lo, hi] with `bins` equal buckets.
# Returns [counts, edges] where edges has bins+1 entries.
fn histogram(xs, bins, lo, hi) {
    int b = 0; int counts = 0; int edges = 0; int i = 0;
    if (lo == nil) { lo = arrayx.minv(xs); }
    if (hi == nil) { hi = arrayx.maxv(xs); }
    if (hi - lo < 1.0e-12) { hi = lo + 1.0; }
    counts = [];
    for (i = 0; i < bins; ++i) { counts[i] = 0; }
    for x in xs {
        b = int(floor((x - lo) / (hi - lo) * bins));
        if (b < 0) { b = 0; }
        if (b >= bins) { b = bins - 1; }
        counts[b] = counts[b] + 1;
    }
    edges = [];
    for (i = 0; i <= bins; ++i) { edges[i] = lo + (hi - lo) * i / bins; }
    return [counts, edges];
}

# Print a quick text histogram.
fn print_histogram(xs, bins, width) {
    int bar = 0; int counts = 0; int edges = 0; int h = 0; int i = 0; int j = 0; int lo2 = 0; int mx = 0; int w = 0;
    if (width == nil) { width = 40; }
    h = histogram(xs, bins, nil, nil);
    counts = h[0];
    edges = h[1];
    mx = arrayx.maxv(counts);
    if (mx == 0) { mx = 1; }
    for (i = 0; i < bins; ++i) {
        bar = "";
        w = int(floor(counts[i] * width / mx));
        for (j = 0; j < w; ++j) { bar = bar + "#"; }
        lo2 = edges[i];
        print(lo2, "\t", bar, " ", counts[i]);
    }
}

# Simple summary array: [count, mean, std, min, q25, median, q75, max].
fn describe(xs) {
    return [len(xs), mean(xs), stddev(xs, 0), arrayx.minv(xs),
            quantile(xs, 0.25), median(xs), quantile(xs, 0.75), arrayx.maxv(xs)];
}

# Welch's t statistic between two samples: [t, dof].
fn welch_t(a, b) {
    int den = 0; int dof = 0; int na = 0; int nb = 0; int num = 0; int se = 0; int t = 0; int va = 0; int vb = 0;
    na = len(a);
    nb = len(b);
    va = variance(a, 1);
    vb = variance(b, 1);
    se = sqrt(va / na + vb / nb);
    if (se < 1.0e-12) { return [0.0, na + nb - 2.0]; }
    t = (mean(a) - mean(b)) / se;
    num = (va / na + vb / nb) ** 2;
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1);
    dof = num / mathx.maxv(den, 1.0e-12);
    return [t, dof];
}
