# =============================================================================
# arrayx.em - array utilities for Emerald
#
# Emerald arrays are values: they copy on assignment and when passed to a
# function. Nothing here can mutate your array in place - every "mutator"
# returns a NEW array, so write:   a = arrayx.push(a, v);
#
# Inside your own scope, the fastest append is direct:  a[len(a)] = v;
# =============================================================================

import prng;
# emx-scope-safe

fn zeros(n) {
    int a = 0; int i = 0;
    a = [];
    for (i = 0; i < n; ++i) { a[i] = 0; }
    return a;
}

fn zerosf(n) {
    int a = 0; int i = 0;
    a = [];
    for (i = 0; i < n; ++i) { a[i] = 0.0; }
    return a;
}

fn full(n, v) {
    int a = 0; int i = 0;
    a = [];
    for (i = 0; i < n; ++i) { a[i] = v; }
    return a;
}

# Half-open integer range [start, stop) with optional step.
fn range(start, stop, step) {
    int a = 0; int v = 0;
    if (step == nil) { step = 1; }
    a = [];
    if (step > 0) { for (v = start; v < stop; v = v + step) { a[len(a)] = v; } }
    elif (step < 0) { for (v = start; v > stop; v = v + step) { a[len(a)] = v; } }
    return a;
}

# n evenly spaced floats from a to b inclusive.
fn linspace(a, b, n) {
    int i = 0; int out = 0;
    out = [];
    if (n == 1) { out[0] = a; return out; }
    for (i = 0; i < n; ++i) { out[i] = a + (b - a) * i / (n - 1.0); }
    return out;
}

# Because parameters arrive by value, this already IS a deep copy.
fn copy(a) { return a; }

fn push(a, v) { a[len(a)] = v; return a; }

# Returns [arrayWithoutLast, lastValue].
fn pop(a) {
    int i = 0; int n = 0; int out = 0; int v = 0;
    n = len(a);
    if (n == 0) { return [a, nil]; }
    v = a[n - 1];
    out = [];
    for (i = 0; i < n - 1; ++i) { out[i] = a[i]; }
    return [out, v];
}

# Half-open slice [i, j). Safe when the range is empty or out of bounds.
fn slice(a, i, j) {
    int k = 0; int n = 0; int out = 0;
    n = len(a);
    if (i < 0) { i = 0; }
    if (j == nil || j > n) { j = n; }
    out = [];
    for (k = i; k < j; ++k) { out[len(out)] = a[k]; }
    return out;
}

fn take(a, n) { return slice(a, 0, n); }
fn drop(a, n) { return slice(a, n, len(a)); }

fn reverse(a) {
    int i = 0; int n = 0; int out = 0;
    out = [];
    n = len(a);
    for (i = 0; i < n; ++i) { out[i] = a[n - 1 - i]; }
    return out;
}

fn insert_at(a, idx, v) {
    int i = 0; int n = 0; int out = 0;
    out = [];
    n = len(a);
    for (i = 0; i < idx; ++i) { out[len(out)] = a[i]; }
    out[len(out)] = v;
    for (i = idx; i < n; ++i) { out[len(out)] = a[i]; }
    return out;
}

fn remove_at(a, idx) {
    int i = 0; int n = 0; int out = 0;
    out = [];
    n = len(a);
    for (i = 0; i < n; ++i) { if (i != idx) { out[len(out)] = a[i]; } }
    return out;
}

fn swap(a, i, j) {
    int t = 0; t = a[i]; a[i] = a[j]; a[j] = t; return a; }

fn contains(a, v) {
    for x in a { if (x == v) { return true; } }
    return false;
}

fn index_of(a, v) {
    int i = 0;
    for (i = 0; i < len(a); ++i) { if (a[i] == v) { return i; } }
    return -1;
}

fn count_of(a, v) {
    int c = 0;
    c = 0;
    for x in a { if (x == v) { c = c + 1; } }
    return c;
}

fn sumv(a) {
    int s = 0;
    s = 0.0;
    for x in a { s = s + x; }
    return s;
}

fn prodv(a) {
    int s = 0;
    s = 1.0;
    for x in a { s = s * x; }
    return s;
}

fn minv(a) {
    int m = 0;
    m = a[0];
    for x in a { if (x < m) { m = x; } }
    return m;
}

fn maxv(a) {
    int m = 0;
    m = a[0];
    for x in a { if (x > m) { m = x; } }
    return m;
}

fn argmin(a) {
    int bi = 0; int i = 0;
    bi = 0;
    for (i = 1; i < len(a); ++i) { if (a[i] < a[bi]) { bi = i; } }
    return bi;
}

fn argmax(a) {
    int bi = 0; int i = 0;
    bi = 0;
    for (i = 1; i < len(a); ++i) { if (a[i] > a[bi]) { bi = i; } }
    return bi;
}

fn flatten1(a) {
    int out = 0;
    out = [];
    for x in a {
        if (type(x) == "array") { for y in x { out[len(out)] = y; } }
        else { out[len(out)] = x; }
    }
    return out;
}

fn repeat(a, times) {
    int out = 0; int t = 0;
    out = [];
    for (t = 0; t < times; ++t) { for x in a { out[len(out)] = x; } }
    return out;
}

fn arr_eq(a, b) {
    int i = 0;
    if (len(a) != len(b)) { return false; }
    for (i = 0; i < len(a); ++i) { if (!(a[i] == b[i])) { return false; } }
    return true;
}

# ---------------------------------------------------------------------------
# Higher-order helpers (Emerald functions are first-class values).
# ---------------------------------------------------------------------------

fn map(a, f) {
    int i = 0; int out = 0;
    out = [];
    for (i = 0; i < len(a); ++i) { out[i] = f(a[i]); }
    return out;
}

fn filter(a, pred) {
    int out = 0;
    out = [];
    for x in a { if (pred(x)) { out[len(out)] = x; } }
    return out;
}

fn reduce(a, f, init) {
    int acc = 0;
    acc = init;
    for x in a { acc = f(acc, x); }
    return acc;
}

fn any_of(a, pred) {
    for x in a { if (pred(x)) { return true; } }
    return false;
}

fn all_of(a, pred) {
    for x in a { if (!pred(x)) { return false; } }
    return true;
}

# ---------------------------------------------------------------------------
# Sorting. In-place quicksort on a local copy, median-of-three pivot so that
# already-sorted input stays at O(log n) recursion depth (Emerald's call
# stack is finite), insertion sort for small partitions.
# ---------------------------------------------------------------------------

fn qs_part(a, lo, hi) {
    int i = 0; int j = 0; int mid = 0; int piv = 0;
    mid = (lo + hi) // 2;
    # Median-of-three: order a[lo], a[mid], a[hi].
    if (a[mid] < a[lo]) { a = swap(a, mid, lo); }
    if (a[hi] < a[lo]) { a = swap(a, hi, lo); }
    if (a[hi] < a[mid]) { a = swap(a, hi, mid); }
    piv = a[mid];
    i = lo;
    j = hi;
    while (i <= j) {
        while (a[i] < piv) { i = i + 1; }
        while (a[j] > piv) { j = j - 1; }
        if (i <= j) { a = swap(a, i, j); i = i + 1; j = j - 1; }
    }
    return [a, i, j];
}

fn qs_rec(a, lo, hi) {
    int i = 0; int j = 0; int r = 0; int v = 0;
    while (lo < hi) {
        if (hi - lo < 12) {
            for (i = lo + 1; i <= hi; ++i) {
                v = a[i];
                j = i - 1;
                while (j >= lo && a[j] > v) { a[j + 1] = a[j]; j = j - 1; }
                a[j + 1] = v;
            }
            return a;
        }
        r = qs_part(a, lo, hi);
        a = r[0];
        i = r[1];
        j = r[2];
        # Recurse into the smaller side, loop on the larger: depth O(log n).
        if (j - lo < hi - i) { a = qs_rec(a, lo, j); lo = i; }
        else { a = qs_rec(a, i, hi); hi = j; }
    }
    return a;
}

fn sort(a) {
    if (len(a) < 2) { return a; }
    return qs_rec(a, 0, len(a) - 1);
}

fn sort_desc(a) { return reverse(sort(a)); }

# Indices that would sort `vals` ascending.
fn argsort(vals) {
    int i = 0; int idx = 0; int n = 0;
    n = len(vals);
    idx = [];
    for (i = 0; i < n; ++i) { idx[i] = i; }
    idx = argsort_rec(idx, vals, 0, n - 1);
    return idx;
}

fn argsort_rec(idx, vals, lo, hi) {
    int i = 0; int j = 0; int mid = 0; int piv = 0; int v = 0;
    while (lo < hi) {
        if (hi - lo < 12) {
            for (i = lo + 1; i <= hi; ++i) {
                v = idx[i];
                j = i - 1;
                while (j >= lo && vals[idx[j]] > vals[v]) { idx[j + 1] = idx[j]; j = j - 1; }
                idx[j + 1] = v;
            }
            return idx;
        }
        mid = (lo + hi) // 2;
        if (vals[idx[mid]] < vals[idx[lo]]) { idx = swap(idx, mid, lo); }
        if (vals[idx[hi]] < vals[idx[lo]]) { idx = swap(idx, hi, lo); }
        if (vals[idx[hi]] < vals[idx[mid]]) { idx = swap(idx, hi, mid); }
        piv = vals[idx[mid]];
        i = lo;
        j = hi;
        while (i <= j) {
            while (vals[idx[i]] < piv) { i = i + 1; }
            while (vals[idx[j]] > piv) { j = j - 1; }
            if (i <= j) { idx = swap(idx, i, j); i = i + 1; j = j - 1; }
        }
        if (j - lo < hi - i) { idx = argsort_rec(idx, vals, lo, j); lo = i; }
        else { idx = argsort_rec(idx, vals, i, hi); hi = j; }
    }
    return idx;
}

# Reorder a by an index array: out[k] = a[order[k]].
fn gather(a, order) {
    int k = 0; int out = 0;
    out = [];
    for (k = 0; k < len(order); ++k) { out[k] = a[order[k]]; }
    return out;
}

# Sorted unique values of a numeric array.
fn unique(a) {
    int i = 0; int out = 0; int s = 0;
    if (len(a) == 0) { return []; }
    s = sort(a);
    out = [s[0]];
    for (i = 1; i < len(s); ++i) {
        if (!(s[i] == s[i - 1])) { out[len(out)] = s[i]; }
    }
    return out;
}

# Fisher-Yates shuffle with explicit seed. Returns [shuffled, newSeed].
fn shuffle(a, seed) {
    int i = 0; int n = 0; int r = 0;
    n = len(a);
    for (i = n - 1; i > 0; --i) {
        r = prng.randint(seed, 0, i);
        seed = r[1];
        a = swap(a, i, r[0]);
    }
    return [a, seed];
}

# k distinct elements sampled without replacement. Returns [sample, newSeed].
fn sample(a, k, seed) {
    int r = 0;
    r = shuffle(a, seed);
    return [slice(r[0], 0, k), r[1]];
}
