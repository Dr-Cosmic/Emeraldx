int LCG_A = 0;
int LCG_M = 0;
# emx-scope-safe
# =============================================================================
# prng.em - deterministic pseudo-random numbers for Emerald
#
# Emerald has no builtin RNG, and everything is passed by value, so the
# functional API threads the generator state explicitly:
#
#     s = prng.seed_from(42);
#     r = prng.randf(s);  x = r[0];  s = r[1];
#
# Every function that consumes randomness returns the advanced state so the
# whole library is reproducible. A convenience Rng class (mutating its own
# field) is provided for interactive use.
#
# Generator: MINSTD (Park-Miller) - state = state * 48271 mod (2^31 - 1).
# All intermediates stay far below 2^53, so it is safe even on float backends.
# =============================================================================

LCG_M = 2147483647;
LCG_A = 48271;

# Normalize any integer into a valid nonzero state in [1, LCG_M - 1].
fn seed_from(x) {
    int s = 0;
    s = int(x) % LCG_M;
    if (s < 0) { s = s + LCG_M; }
    if (s == 0) { s = 1; }
    # One warm-up step decorrelates small consecutive seeds.
    return (s * LCG_A) % LCG_M;
}

# Advance the state one step.
fn lcg(s) {
    s = (int(s) * LCG_A) % LCG_M;
    if (s <= 0) { s = 1; }
    return s;
}

# Uniform float in [0, 1). Returns [value, newState].
fn randf(s) {
    s = lcg(s);
    return [(s - 1) / 2147483646.0, s];
}

# Uniform float in [lo, hi). Returns [value, newState].
fn uniform(s, lo, hi) {
    int r = 0;
    r = randf(s);
    return [lo + (hi - lo) * r[0], r[1]];
}

# Uniform integer in [lo, hi] inclusive. Returns [value, newState].
fn randint(s, lo, hi) {
    int n = 0; int r = 0;
    r = randf(s);
    n = int(floor(r[0] * (hi - lo + 1)));
    if (n > hi - lo) { n = hi - lo; }
    return [lo + n, r[1]];
}

# Bernoulli trial with probability p. Returns [true/false, newState].
fn chance(s, p) {
    int r = 0;
    r = randf(s);
    return [r[0] < p, r[1]];
}

# Standard normal via Box-Muller. Returns [z, newState].
fn normal(s) {
    int r1 = 0; int r2 = 0; int u1 = 0; int u2 = 0; int z = 0;
    r1 = randf(s);
    u1 = r1[0];
    r2 = randf(r1[1]);
    u2 = r2[0];
    if (u1 < 1.0e-12) { u1 = 1.0e-12; }
    z = sqrt(-2.0 * ln(u1)) * cos(6.283185307179586 * u2);
    return [z, r2[1]];
}

# Normal with mean mu and std sigma. Returns [value, newState].
fn gauss(s, mu, sigma) {
    int r = 0;
    r = normal(s);
    return [mu + sigma * r[0], r[1]];
}

# Array of n uniform floats in [lo, hi). Returns [array, newState].
fn randf_array(s, n, lo, hi) {
    int i = 0; int out = 0; int r = 0;
    if (lo == nil) { lo = 0.0; }
    if (hi == nil) { hi = 1.0; }
    out = [];
    for (i = 0; i < n; ++i) {
        r = randf(s);
        out[i] = lo + (hi - lo) * r[0];
        s = r[1];
    }
    return [out, s];
}

# Array of n standard normals. Returns [array, newState].
fn normal_array(s, n) {
    int i = 0; int out = 0; int r = 0;
    out = [];
    for (i = 0; i < n; ++i) {
        r = normal(s);
        out[i] = r[0];
        s = r[1];
    }
    return [out, s];
}

# Pick one element of arr. Returns [element, newState].
fn choice(s, arr) {
    int r = 0;
    r = randint(s, 0, len(arr) - 1);
    return [arr[r[0]], r[1]];
}

# Sample an index from a discrete (unnormalized) weight array.
# Returns [index, newState].
fn weighted_index(s, weights) {
    int acc = 0; int i = 0; int r = 0; int target = 0; int total = 0;
    total = 0.0;
    for w in weights { total = total + w; }
    if (total <= 0) { return [0, lcg(s)]; }
    r = randf(s);
    target = r[0] * total;
    acc = 0.0;
    for (i = 0; i < len(weights); ++i) {
        acc = acc + weights[i];
        if (target < acc) { return [i, r[1]]; }
    }
    return [len(weights) - 1, r[1]];
}

# Convenience stateful generator. Methods mutate the internal state field.
# Prefer the functional API inside libraries; this is for scripts and REPL.
class Rng(seedInit) {
    emxState = seedInit;
    fn step() {
        # Python-style % keeps this non-negative even for negative seeds.
        emxState = (emxState * 48271) % 2147483647;
        if (emxState == 0) { emxState = 1; }
        return emxState;
    }
    fn nextf() { return (step() - 1) / 2147483646.0; }
    fn nexti(lo, hi) {
    int n = 0;
        n = int(floor(nextf() * (hi - lo + 1)));
        if (n > hi - lo) { n = hi - lo; }
        return lo + n;
    }
    fn gaussf() {
    int u1 = 0; int u2 = 0;
        u1 = nextf();
        u2 = nextf();
        if (u1 < 1.0e-12) { u1 = 1.0e-12; }
        return sqrt(-2.0 * ln(u1)) * cos(6.283185307179586 * u2);
    }
}
