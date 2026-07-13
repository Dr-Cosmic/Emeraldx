# emx-scope-safe
# =============================================================================
# dictx.em - a small dictionary for Emerald
#
# Represented as d = [keysArray, valuesArray]. Keys are compared with the
# loose == (strings and numbers both work). Lookups are linear scans, which
# is fine for the vocabulary/cache sizes this library uses. Like everything
# in Emerald, dictionaries are values:   d = dictx.dset(d, k, v);
# =============================================================================

fn dnew() { return [[], []]; }

fn dsize(d) { return len(d[0]); }

fn dfind(d, k) {
    int i = 0; int ks = 0;
    ks = d[0];
    for (i = 0; i < len(ks); ++i) { if (ks[i] == k) { return i; } }
    return -1;
}

fn dhas(d, k) { return dfind(d, k) >= 0; }

fn dget(d, k, default) {
    int i = 0; int vs = 0;
    i = dfind(d, k);
    if (i < 0) { return default; }
    vs = d[1];
    return vs[i];
}

fn dset(d, k, v) {
    int i = 0; int ks = 0; int vs = 0;
    i = dfind(d, k);
    ks = d[0];
    vs = d[1];
    if (i < 0) {
        ks[len(ks)] = k;
        vs[len(vs)] = v;
    } else {
        vs[i] = v;
    }
    return [ks, vs];
}

# Add `by` (default 1) to a numeric entry, creating it at 0 first.
fn dinc(d, k, by) {
    int i = 0; int ks = 0; int vs = 0;
    if (by == nil) { by = 1; }
    i = dfind(d, k);
    ks = d[0];
    vs = d[1];
    if (i < 0) {
        ks[len(ks)] = k;
        vs[len(vs)] = by;
    } else {
        vs[i] = vs[i] + by;
    }
    return [ks, vs];
}

fn ddel(d, k) {
    int i = 0; int j = 0; int ks = 0; int nk = 0; int nv = 0; int vs = 0;
    i = dfind(d, k);
    if (i < 0) { return d; }
    ks = d[0];
    vs = d[1];
    nk = [];
    nv = [];
    for (j = 0; j < len(ks); ++j) {
        if (j != i) {
            nk[len(nk)] = ks[j];
            nv[len(nv)] = vs[j];
        }
    }
    return [nk, nv];
}

fn dkeys(d) { return d[0]; }
fn dvals(d) { return d[1]; }

# Key of the largest numeric value (nil for an empty dict).
fn dargmax(d) {
    int bi = 0; int i = 0; int ks = 0; int vs = 0;
    ks = d[0];
    vs = d[1];
    if (len(ks) == 0) { return nil; }
    bi = 0;
    for (i = 1; i < len(vs); ++i) { if (vs[i] > vs[bi]) { bi = i; } }
    return ks[bi];
}

# [keys, values] sorted by value descending (selection sort; small dicts).
fn dsorted_by_value_desc(d) {
    int bi = 0; int i = 0; int j = 0; int ks = 0; int n = 0; int tk = 0; int tv = 0; int vs = 0;
    ks = d[0];
    vs = d[1];
    n = len(ks);
    for (i = 0; i < n; ++i) {
        bi = i;
        for (j = i + 1; j < n; ++j) { if (vs[j] > vs[bi]) { bi = j; } }
        if (bi != i) {
            tv = vs[i]; vs[i] = vs[bi]; vs[bi] = tv;
            tk = ks[i]; ks[i] = ks[bi]; ks[bi] = tk;
        }
    }
    return [ks, vs];
}
