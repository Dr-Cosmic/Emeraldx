# =============================================================================
# complexx.em - complex numbers and the Fast Fourier Transform
#
# A complex number is [re, im]. Signals are passed as separate re[]/im[]
# arrays for speed. FFT is iterative radix-2 (input padded to a power of 2),
# so no recursion depth worries.
# =============================================================================

import mathx;
import arrayx;
# emx-scope-safe

fn cnum(re, im) {
    if (im == nil) { im = 0.0; }
    return [re, im];
}

fn cadd(a, b) { return [a[0] + b[0], a[1] + b[1]]; }
fn csub(a, b) { return [a[0] - b[0], a[1] - b[1]]; }

fn cmul(a, b) {
    return [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
}

fn cdiv(a, b) {
    int d = 0;
    d = b[0] * b[0] + b[1] * b[1];
    if (d < 1.0e-300) { return [0.0, 0.0]; }
    return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d];
}

fn cconj(a) { return [a[0], 0.0 - a[1]]; }
fn cabs(a) { return sqrt(a[0] * a[0] + a[1] * a[1]); }
fn carg(a) { return mathx.atan2(a[1], a[0]); }
fn cscale(a, s) { return [a[0] * s, a[1] * s]; }

fn from_polar(r, theta) { return [r * cos(theta), r * sin(theta)]; }

fn cexp(a) {
    int m = 0;
    m = exp(a[0]);
    return [m * cos(a[1]), m * sin(a[1])];
}

# Principal complex power a^p for real p.
fn cpow(a, p) {
    int r = 0; int th = 0;
    r = cabs(a);
    if (r < 1.0e-300) { return [0.0, 0.0]; }
    th = carg(a);
    return from_polar(r ** p, th * p);
}

fn csqrt(a) { return cpow(a, 0.5); }

# ---------------------------------------------------------------------------
# FFT
# ---------------------------------------------------------------------------

fn next_pow2(n) {
    int p = 0;
    p = 1;
    while (p < n) { p = p * 2; }
    return p;
}

fn pad_to(arr, n, fill) {
    int i = 0; int out = 0;
    if (fill == nil) { fill = 0.0; }
    out = [];
    for (i = 0; i < n; ++i) {
        if (i < len(arr)) { out[i] = arr[i]; }
        else { out[i] = fill; }
    }
    return out;
}

# Bit-reversal permutation index (computed arithmetically; Emerald has no
# bitwise operators).
fn bit_reverse(i, bits) {
    int b = 0; int r = 0;
    r = 0;
    for (b = 0; b < bits; ++b) {
        r = r * 2 + i % 2;
        i = i // 2;
    }
    return r;
}

# In-place style iterative radix-2 FFT on local copies.
# invert = true computes the inverse transform (scaled by 1/n).
# Returns [re, im]; input length must be a power of two (use fft_any).
fn fft_core(reIn, imIn, invert) {
    int ang = 0; int bits = 0; int cwi = 0; int cwr = 0; int half = 0; int i = 0; int i0 = 0; int i1 = 0; int im = 0; int j = 0; int k = 0; int n = 0; int nwr = 0; int re = 0; int start = 0; int t = 0; int ti = 0; int tr = 0; int ui = 0; int ur = 0; int vi = 0; int vr = 0; int wi = 0; int wr = 0;
    re = reIn;
    im = imIn;
    n = len(re);
    bits = 0;
    t = n;
    while (t > 1) { bits = bits + 1; t = t // 2; }
    for (i = 0; i < n; ++i) {
        j = bit_reverse(i, bits);
        if (i < j) {
            tr = re[i]; re[i] = re[j]; re[j] = tr;
            ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    length = 2;
    while (length <= n) {
        ang = mathx.TAU / length;
        if (!invert) { ang = 0.0 - ang; }
        wr = cos(ang);
        wi = sin(ang);
        half = length // 2;
        for (start = 0; start < n; start = start + length) {
            cwr = 1.0;
            cwi = 0.0;
            for (k = 0; k < half; ++k) {
                i0 = start + k;
                i1 = start + k + half;
                ur = re[i0]; ui = im[i0];
                vr = re[i1] * cwr - im[i1] * cwi;
                vi = re[i1] * cwi + im[i1] * cwr;
                re[i0] = ur + vr; im[i0] = ui + vi;
                re[i1] = ur - vr; im[i1] = ui - vi;
                nwr = cwr * wr - cwi * wi;
                cwi = cwr * wi + cwi * wr;
                cwr = nwr;
            }
        }
        length = length * 2;
    }
    if (invert) {
        for (i = 0; i < n; ++i) { re[i] = re[i] / n; im[i] = im[i] / n; }
    }
    return [re, im];
}

# FFT of a real or complex signal of ANY length (zero-padded to power of 2).
# Returns [re, im].
fn fft(reIn, imIn) {
    int n = 0;
    n = next_pow2(len(reIn));
    if (imIn == nil) { imIn = []; }
    return fft_core(pad_to(reIn, n, 0.0), pad_to(imIn, n, 0.0), false);
}

fn ifft(reIn, imIn) {
    int n = 0;
    n = next_pow2(len(reIn));
    return fft_core(pad_to(reIn, n, 0.0), pad_to(imIn, n, 0.0), true);
}

# Magnitude spectrum of a real signal. Returns the first n/2+1 bins.
fn spectrum(signal) {
    int f = 0; int half = 0; int i = 0; int im = 0; int n = 0; int out = 0; int re = 0;
    f = fft(signal, nil);
    re = f[0];
    im = f[1];
    n = len(re);
    out = [];
    half = n // 2;
    for (i = 0; i <= half; ++i) {
        out[i] = sqrt(re[i] * re[i] + im[i] * im[i]);
    }
    return out;
}

# Linear convolution of two real signals via FFT. Output length na + nb - 1.
fn convolve_fft(a, b) {
    int ai = 0; int ar = 0; int bi = 0; int br = 0; int ci = 0; int cr = 0; int fa = 0; int fb = 0; int i = 0; int inv = 0; int n = 0; int na = 0; int nb = 0; int out = 0; int rr = 0;
    na = len(a);
    nb = len(b);
    n = next_pow2(na + nb - 1);
    fa = fft_core(pad_to(a, n, 0.0), arrayx.zerosf(n), false);
    fb = fft_core(pad_to(b, n, 0.0), arrayx.zerosf(n), false);
    ar = fa[0]; ai = fa[1];
    br = fb[0]; bi = fb[1];
    cr = [];
    ci = [];
    for (i = 0; i < n; ++i) {
        cr[i] = ar[i] * br[i] - ai[i] * bi[i];
        ci[i] = ar[i] * bi[i] + ai[i] * br[i];
    }
    inv = fft_core(cr, ci, true);
    rr = inv[0];
    out = [];
    for (i = 0; i < na + nb - 1; ++i) { out[i] = rr[i]; }
    return out;
}

# Direct O(n*m) convolution for short kernels / verification.
fn convolve_direct(a, b) {
    int av = 0; int i = 0; int j = 0; int na = 0; int nb = 0; int out = 0;
    na = len(a);
    nb = len(b);
    out = arrayx.zerosf(na + nb - 1);
    for (i = 0; i < na; ++i) {
        av = a[i];
        if (av != 0.0) {
            for (j = 0; j < nb; ++j) { out[i + j] = out[i + j] + av * b[j]; }
        }
    }
    return out;
}

# Naive DFT (for testing the FFT).
fn dft_naive(signal) {
    int ang = 0; int im = 0; int k = 0; int n = 0; int re = 0; int si = 0; int sr = 0; int t = 0;
    n = len(signal);
    re = [];
    im = [];
    for (k = 0; k < n; ++k) {
        sr = 0.0;
        si = 0.0;
        for (t = 0; t < n; ++t) {
            ang = 0.0 - mathx.TAU * k * t / n;
            sr = sr + signal[t] * cos(ang);
            si = si + signal[t] * sin(ang);
        }
        re[k] = sr;
        im[k] = si;
    }
    return [re, im];
}

# ---------------------------------------------------------------------------
# Windows and simple signal helpers
# ---------------------------------------------------------------------------

fn hann_window(n) {
    int i = 0; int w = 0;
    w = [];
    for (i = 0; i < n; ++i) { w[i] = 0.5 - 0.5 * cos(mathx.TAU * i / (n - 1.0)); }
    return w;
}

fn hamming_window(n) {
    int i = 0; int w = 0;
    w = [];
    for (i = 0; i < n; ++i) { w[i] = 0.54 - 0.46 * cos(mathx.TAU * i / (n - 1.0)); }
    return w;
}

fn apply_window(signal, w) {
    int i = 0; int out = 0;
    out = [];
    for (i = 0; i < len(signal); ++i) { out[i] = signal[i] * w[i]; }
    return out;
}

# Sample a sine wave: n samples, frequency in cycles per full buffer.
fn sine_wave(n, cycles, amplitude, phase) {
    int i = 0; int out = 0;
    if (amplitude == nil) { amplitude = 1.0; }
    if (phase == nil) { phase = 0.0; }
    out = [];
    for (i = 0; i < n; ++i) {
        out[i] = amplitude * sin(mathx.TAU * cycles * i / n + phase);
    }
    return out;
}
