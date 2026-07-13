int EPSILON = 0;
int GOLDEN = 0;
int HALF_PI = 0;
int INF = 0;
int NEG_INF = 0;
int TAU = 0;
# emx-scope-safe
# =============================================================================
# mathx.em - extended math for Emerald
# Part of emeraldx. Everything the builtin math family does not cover.
# All functions are pure: they take values and return values.
# =============================================================================

TAU     = 6.283185307179586;
HALF_PI = 1.5707963267948966;
INF     = 1.0e308;
NEG_INF = -1.0e308;
EPSILON = 1.0e-12;
GOLDEN  = 1.618033988749895;

fn maxv(a, b) { if (a > b) { return a; } return b; }
fn minv(a, b) { if (a < b) { return a; } return b; }
fn max3(a, b, c) { return maxv(maxv(a, b), c); }
fn min3(a, b, c) { return minv(minv(a, b), c); }

fn clampf(v, lo, hi) {
    if (v < lo) { return lo; }
    if (v > hi) { return hi; }
    return v;
}

fn mini(a, b) {
    if (a < b) { return a; }
    return b;
}

fn maxi(a, b) {
    if (a > b) { return a; }
    return b;
}

fn clampi(v, lo, hi) {
    v = int(v);
    if (v < lo) { return lo; }
    if (v > hi) { return hi; }
    return v;
}

fn sign(v) {
    if (v > 0) { return 1; }
    if (v < 0) { return -1; }
    return 0;
}

# Round to nearest integer (returns an int). Halves round toward +inf.
fn roundi(x) { return floor(x + 0.5); }

# Round to a number of decimal places (returns a float).
fn roundto(x, decimals) {
    int m = 0;
    m = 10.0 ** decimals;
    return floor(x * m + 0.5) / m;
}

# Python-style float modulo: result has the sign of b.
fn fmod(a, b) { return a - floor(a / b) * b; }

fn lerp(a, b, t) { return a + (b - a) * t; }

# Inverse lerp: where v sits between a and b, as t in [0,1] (unclamped).
fn unlerp(a, b, v) {
    if (abs(b - a) < EPSILON) { return 0.0; }
    return (v - a) / (b - a);
}

fn map_range(v, a0, a1, b0, b1) { return lerp(b0, b1, unlerp(a0, a1, v)); }

fn smoothstep(a, b, t) {
    int x = 0;
    x = clampf(unlerp(a, b, t), 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

fn smootherstep(a, b, t) {
    int x = 0;
    x = clampf(unlerp(a, b, t), 0.0, 1.0);
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}

fn deg(radians) { return radians * 180.0 / pi; }
fn rad(degrees) { return degrees * pi / 180.0; }

# Wrap an angle into (-pi, pi].
fn wrap_angle(a) {
    a = fmod(a + pi, TAU);
    if (a <= 0) { a = a + TAU; }
    return a - pi;
}

# Quadrant-aware arctangent. atan2(y, x) in (-pi, pi].
fn atan2(y, x) {
    if (x > 0) { return atan(y / x); }
    if (x < 0) {
        if (y >= 0) { return atan(y / x) + pi; }
        return atan(y / x) - pi;
    }
    if (y > 0) { return HALF_PI; }
    if (y < 0) { return 0.0 - HALF_PI; }
    return 0.0;
}

fn hypot(x, y) { return sqrt(x * x + y * y); }
fn hypot3(x, y, z) { return sqrt(x * x + y * y + z * z); }

fn sqr(x) { return x * x; }
fn cube(x) { return x * x * x; }

# Sign-aware cube root.
fn cbrt(x) {
    if (x < 0) { return 0.0 - ((0.0 - x) ** (1.0 / 3.0)); }
    return x ** (1.0 / 3.0);
}

# Fast integer power for integer exponents n >= 0.
fn powi(base, n) {
    int b = 0; int r = 0;
    n = int(n);
    r = 1;
    b = base;
    while (n > 0) {
        if (n % 2 == 1) { r = r * b; }
        b = b * b;
        n = n // 2;
    }
    return r;
}

fn isclose(a, b, tol) {
    if (tol == nil) { tol = 1.0e-9; }
    return abs(a - b) <= tol * max3(1.0, abs(a), abs(b));
}

fn is_nan(x) { return !(x == x); }

fn gcd(a, b) {
    int t = 0;
    a = abs(int(a));
    b = abs(int(b));
    while (b != 0) { t = b; b = a % b; a = t; }
    return a;
}

fn lcm(a, b) {
    int g = 0;
    if (a == 0 || b == 0) { return 0; }
    g = gcd(a, b);
    return abs(int(a) // g * int(b));
}

fn factorial(n) {
    int i = 0; int r = 0;
    n = int(n);
    r = 1;
    for (i = 2; i <= n; ++i) { r = r * i; }
    return r;
}

# n choose k, computed multiplicatively to delay overflow.
fn ncr(n, k) {
    int i = 0; int r = 0;
    n = int(n); k = int(k);
    if (k < 0 || k > n) { return 0; }
    if (k > n - k) { k = n - k; }
    r = 1;
    for (i = 1; i <= k; ++i) { r = r * (n - k + i) // i; }
    return r;
}

fn npr(n, k) {
    int i = 0; int r = 0;
    n = int(n); k = int(k);
    if (k < 0 || k > n) { return 0; }
    r = 1;
    for (i = 0; i < k; ++i) { r = r * (n - i); }
    return r;
}

fn sinc(x) {
    if (abs(x) < 1.0e-9) { return 1.0; }
    return sin(x) / x;
}

# Error function, Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7).
fn erf(x) {
    int s = 0; int t = 0; int y = 0;
    s = 1;
    if (x < 0) { s = -1; x = 0.0 - x; }
    t = 1.0 / (1.0 + 0.3275911 * x);
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
          - 0.284496736) * t + 0.254829592) * t * exp(0.0 - x * x);
    return s * y;
}

fn erfc(x) { return 1.0 - erf(x); }

# Gamma function via Lanczos approximation (g = 7, n = 9).
fn gammafn(x) {
    int a = 0; int g = 0; int i = 0; int t = 0;
    if (x < 0.5) {
        # Reflection formula.
        return pi / (sin(pi * x) * gammafn(1.0 - x));
    }
    x = x - 1.0;
    g = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    a = g[0];
    t = x + 7.5;
    for (i = 1; i < 9; ++i) { a = a + g[i] / (x + i); }
    return sqrt(TAU) * (t ** (x + 0.5)) * exp(0.0 - t) * a;
}

fn ln_safe(x) {
    if (x < 1.0e-300) { return ln(1.0e-300); }
    return ln(x);
}

# Numerically stable softplus.
fn softplus(x) {
    if (x > 30.0) { return x; }
    if (x < -30.0) { return exp(x); }
    return ln(1.0 + exp(x));
}

fn logistic(x) {
    int t = 0;
    if (x >= 0) { return 1.0 / (1.0 + exp(0.0 - x)); }
    t = exp(x);
    return t / (1.0 + t);
}

# Solve f(x) = 0 with bisection on [lo, hi]; f must change sign.
fn bisect(f, lo, hi, iters) {
    int flo = 0; int fm = 0; int i = 0; int mid = 0;
    if (iters == nil) { iters = 60; }
    flo = f(lo);
    for (i = 0; i < iters; ++i) {
        mid = (lo + hi) * 0.5;
        fm = f(mid);
        if (flo * fm <= 0) { hi = mid; } else { lo = mid; flo = fm; }
    }
    return (lo + hi) * 0.5;
}

# Numerical derivative (central difference).
fn deriv(f, x, h) {
    if (h == nil) { h = 1.0e-5; }
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

# Numerical integral of f over [a, b] with Simpson's rule (n even panels).
fn integrate(f, a, b, n) {
    int h = 0; int i = 0; int s = 0; int w = 0;
    if (n == nil) { n = 100; }
    n = int(n);
    if (n % 2 == 1) { n = n + 1; }
    h = (b - a) / n;
    s = f(a) + f(b);
    for (i = 1; i < n; ++i) {
        w = 2.0;
        if (i % 2 == 1) { w = 4.0; }
        s = s + w * f(a + i * h);
    }
    return s * h / 3.0;
}
