# =============================================================================
# photo.em - computational photography
#
#   exposure, gamma_correct, reinhard_tonemap        tone curves
#   merge_exposures                                  simple HDR fusion
#   gray_world_wb, scale_channels                    white balance
#   unsharp_mask                                     detail boost
#   bilateral_filter, median_filter                  edge-aware denoising
#   vignette, film_grain                             looks
#   bayer_mosaic, demosaic_bilinear                  sensor simulation
# =============================================================================

import mathx;
import arrayx;
import prng;
import image;
import imgproc;
# emx-scope-safe

# ---------------------------------------------------------------------------
# Tone curves
# ---------------------------------------------------------------------------

# Multiply exposure by 2^stops.
fn exposure(img, stops) {
    int d = 0; int g = 0; int i = 0; int out = 0;
    g = 2.0 ** stops;
    d = img[3];
    out = [];
    for (i = 0; i < len(d); ++i) { out[i] = d[i] * g; }
    return [img[0], img[1], img[2], out];
}

fn gamma_correct(img, gamma) {
    int d = 0; int i = 0; int inv = 0; int out = 0; int v = 0;
    d = img[3];
    inv = 1.0 / gamma;
    out = [];
    for (i = 0; i < len(d); ++i) {
        v = d[i];
        if (v < 0.0) { v = 0.0; }
        out[i] = v ** inv;
    }
    return [img[0], img[1], img[2], out];
}

# Reinhard global operator: L' = L(1 + L/Lw^2) / (1 + L), applied per channel
# after key scaling. Maps HDR (possibly > 1) into [0, 1).
fn reinhard_tonemap(img, key, whitePoint) {
    int d = 0; int i = 0; int l = 0; int lavg = 0; int n = 0; int out = 0; int sc = 0; int w2 = 0;
    if (key == nil) { key = 0.18; }
    if (whitePoint == nil) { whitePoint = 4.0; }
    d = img[3];
    n = len(d);
    # log-average luminance as scene key
    lavg = 0.0;
    for (i = 0; i < n; ++i) { lavg = lavg + ln(1.0e-6 + d[i]); }
    lavg = exp(lavg / n);
    sc = key / mathx.maxv(lavg, 1.0e-6);
    w2 = whitePoint * whitePoint;
    out = [];
    for (i = 0; i < n; ++i) {
        l = d[i] * sc;
        out[i] = l * (1.0 + l / w2) / (1.0 + l);
    }
    return [img[0], img[1], img[2], out];
}

# ---------------------------------------------------------------------------
# HDR merge: images are the SAME scene at different exposures (each already
# in [0,1]); stops[i] gives each image's exposure in stops relative to base.
# Radiance is recovered per pixel with a hat weighting, then averaged.
# Returns a linear HDR image (values may exceed 1) - tonemap afterwards.
# ---------------------------------------------------------------------------

fn merge_exposures(imgs, stops) {
    int acc = 0; int ch = 0; int d = 0; int first = 0; int gain = 0; int h = 0; int i = 0; int img = 0; int k = 0; int n = 0; int out = 0; int v = 0; int w = 0; int wsum = 0; int wt = 0;
    first = imgs[0];
    w = first[0];
    h = first[1];
    ch = first[2];
    n = len(first[3]);
    acc = arrayx.zerosf(n);
    wsum = arrayx.zerosf(n);
    for (k = 0; k < len(imgs); ++k) {
        img = imgs[k];
        d = img[3];
        gain = 2.0 ** stops[k];
        for (i = 0; i < n; ++i) {
            v = d[i];
            # hat weight: trust mid-tones, distrust clipped ends
            wt = 1.0 - abs(2.0 * v - 1.0);
            if (wt < 0.02) { wt = 0.02; }
            acc[i] = acc[i] + wt * v / gain;
            wsum[i] = wsum[i] + wt;
        }
    }
    out = [];
    for (i = 0; i < n; ++i) { out[i] = acc[i] / wsum[i]; }
    return [w, h, ch, out];
}

# ---------------------------------------------------------------------------
# White balance
# ---------------------------------------------------------------------------

fn scale_channels(img, gains) {
    int c = 0; int ch = 0; int d = 0; int h = 0; int i = 0; int out = 0; int w = 0;
    w = img[0];
    h = img[1];
    ch = img[2];
    d = img[3];
    out = [];
    for (i = 0; i < w * h; ++i) {
        for (c = 0; c < ch; ++c) {
            out[i * ch + c] = d[i * ch + c] * gains[c];
        }
    }
    return [w, h, ch, out];
}

# Gray-world assumption: scale channels so their means match the global mean.
fn gray_world_wb(img) {
    int c = 0; int ch = 0; int d = 0; int gains = 0; int h = 0; int i = 0; int means = 0; int total = 0; int w = 0;
    w = img[0];
    h = img[1];
    ch = img[2];
    d = img[3];
    means = arrayx.zerosf(ch);
    for (i = 0; i < w * h; ++i) {
        for (c = 0; c < ch; ++c) { means[c] = means[c] + d[i * ch + c]; }
    }
    total = 0.0;
    for (c = 0; c < ch; ++c) {
        means[c] = means[c] / (w * h);
        total = total + means[c];
    }
    total = total / ch;
    gains = [];
    for (c = 0; c < ch; ++c) {
        gains[c] = total / mathx.maxv(means[c], 1.0e-6);
    }
    return scale_channels(img, gains);
}

# ---------------------------------------------------------------------------
# Detail
# ---------------------------------------------------------------------------

# Unsharp mask on a gray image: out = img + amount * (img - blur).
fn unsharp_mask(img, amount, radius) {
    int b = 0; int blur = 0; int d = 0; int i = 0; int out = 0; int size = 0;
    if (amount == nil) { amount = 1.0; }
    if (radius == nil) { radius = 2; }
    size = radius * 2 + 1;
    blur = imgproc.gaussian_blur(img, size, radius * 0.6);
    d = img[3];
    b = blur[3];
    out = [];
    for (i = 0; i < len(d); ++i) {
        out[i] = image.clamp01(d[i] + amount * (d[i] - b[i]));
    }
    return [img[0], img[1], img[2], out];
}

# ---------------------------------------------------------------------------
# Edge-aware denoising (gray images; run per channel for color)
# ---------------------------------------------------------------------------

fn bilateral_filter(img, radius, sigmaS, sigmaR) {
    int acc = 0; int center = 0; int d = 0; int dr = 0; int dx = 0; int dy = 0; int h = 0; int out = 0; int size = 0; int sw = 0; int twoR2 = 0; int twoS2 = 0; int v = 0; int w = 0; int wacc = 0; int wt = 0; int x = 0; int xx = 0; int y = 0; int yy = 0;
    if (radius == nil) { radius = 2; }
    if (sigmaS == nil) { sigmaS = radius * 0.7; }
    if (sigmaR == nil) { sigmaR = 0.15; }
    w = img[0];
    h = img[1];
    d = img[3];
    twoS2 = 2.0 * sigmaS * sigmaS;
    twoR2 = 2.0 * sigmaR * sigmaR;
    # precompute spatial weights
    size = 2 * radius + 1;
    sw = [];
    for (dy = 0 - radius; dy <= radius; ++dy) {
        for (dx = 0 - radius; dx <= radius; ++dx) {
            sw[(dy + radius) * size + (dx + radius)] =
                exp(0.0 - (dx * dx + dy * dy) / twoS2);
        }
    }
    out = [];
    for (y = 0; y < h; ++y) {
        for (x = 0; x < w; ++x) {
            center = d[y * w + x];
            acc = 0.0;
            wacc = 0.0;
            for (dy = 0 - radius; dy <= radius; ++dy) {
                yy = mathx.clampi(y + dy, 0, h - 1);
                for (dx = 0 - radius; dx <= radius; ++dx) {
                    xx = mathx.clampi(x + dx, 0, w - 1);
                    v = d[yy * w + xx];
                    dr = v - center;
                    wt = sw[(dy + radius) * size + (dx + radius)] *
                         exp(0.0 - dr * dr / twoR2);
                    acc = acc + wt * v;
                    wacc = wacc + wt;
                }
            }
            out[y * w + x] = acc / wacc;
        }
    }
    return [w, h, 1, out];
}

# 3x3 median filter (gray) - excellent against salt & pepper noise.
fn median_filter(img) {
    int a = 0; int b = 0; int d = 0; int dx = 0; int dy = 0; int h = 0; int k = 0; int key = 0; int out = 0; int vals = 0; int w = 0; int x = 0; int xx = 0; int y = 0; int yy = 0;
    w = img[0];
    h = img[1];
    d = img[3];
    out = [];
    vals = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    for (y = 0; y < h; ++y) {
        for (x = 0; x < w; ++x) {
            k = 0;
            for (dy = -1; dy <= 1; ++dy) {
                yy = mathx.clampi(y + dy, 0, h - 1);
                for (dx = -1; dx <= 1; ++dx) {
                    xx = mathx.clampi(x + dx, 0, w - 1);
                    vals[k] = d[yy * w + xx];
                    k = k + 1;
                }
            }
            # insertion sort 9 values, take the middle
            for (a = 1; a < 9; ++a) {
                key = vals[a];
                b = a - 1;
                while (b >= 0 && vals[b] > key) {
                    vals[b + 1] = vals[b];
                    b = b - 1;
                }
                vals[b + 1] = key;
            }
            out[y * w + x] = vals[4];
        }
    }
    return [w, h, 1, out];
}

# ---------------------------------------------------------------------------
# Looks
# ---------------------------------------------------------------------------

# Darken corners: strength 0..1, falloff controls the curve.
fn vignette(img, strength, falloff) {
    int base = 0; int c = 0; int ch = 0; int cx = 0; int cy = 0; int d = 0; int dx = 0; int dy = 0; int f = 0; int h = 0; int maxR = 0; int out = 0; int r = 0; int w = 0; int x = 0; int y = 0;
    if (strength == nil) { strength = 0.5; }
    if (falloff == nil) { falloff = 2.0; }
    w = img[0];
    h = img[1];
    ch = img[2];
    d = img[3];
    cx = (w - 1) / 2.0;
    cy = (h - 1) / 2.0;
    maxR = sqrt(cx * cx + cy * cy);
    out = [];
    for (y = 0; y < h; ++y) {
        for (x = 0; x < w; ++x) {
            dx = (x - cx) / maxR;
            dy = (y - cy) / maxR;
            r = sqrt(dx * dx + dy * dy);
            f = 1.0 - strength * (r ** falloff);
            if (f < 0.0) { f = 0.0; }
            base = (y * w + x) * ch;
            for (c = 0; c < ch; ++c) { out[base + c] = d[base + c] * f; }
        }
    }
    return [w, h, ch, out];
}

# Additive gaussian grain. Returns [img, newSeed].
fn film_grain(img, amount, seed) {
    int d = 0; int i = 0; int noise = 0; int out = 0; int r = 0;
    if (amount == nil) { amount = 0.04; }
    d = img[3];
    r = prng.normal_array(seed, len(d));
    noise = r[0];
    out = [];
    for (i = 0; i < len(d); ++i) {
        out[i] = image.clamp01(d[i] + noise[i] * amount);
    }
    return [[img[0], img[1], img[2], out], r[1]];
}

# ---------------------------------------------------------------------------
# Bayer sensor simulation: RGGB mosaic + bilinear demosaic
# ---------------------------------------------------------------------------

# RGB image -> single-channel RGGB mosaic.
fn bayer_mosaic(img) {
    int b = 0; int c = 0; int d = 0; int h = 0; int out = 0; int w = 0; int x = 0; int y = 0;
    w = img[0];
    h = img[1];
    d = img[3];
    out = [];
    for (y = 0; y < h; ++y) {
        for (x = 0; x < w; ++x) {
            b = (y * w + x) * 3;
            c = 1;                              # green
            if (y % 2 == 0 && x % 2 == 0) { c = 0; }        # red
            elif (y % 2 == 1 && x % 2 == 1) { c = 2; }      # blue
            out[y * w + x] = d[b + c];
        }
    }
    return [w, h, 1, out];
}

fn demosaic_channel_avg(mos, x, y, w, h, wantR, wantC) {
    int acc = 0; int c = 0; int cnt = 0; int d = 0; int dx = 0; int dy = 0; int xx = 0; int yy = 0;
    # average all neighbours (incl. self) in the 3x3 window whose Bayer color
    # matches wantC (0 red, 1 green, 2 blue)
    d = mos[3];
    acc = 0.0;
    cnt = 0;
    for (dy = -1; dy <= 1; ++dy) {
        yy = y + dy;
        if (yy < 0 || yy >= h) { continue; }
        for (dx = -1; dx <= 1; ++dx) {
            xx = x + dx;
            if (xx < 0 || xx >= w) { continue; }
            c = 1;
            if (yy % 2 == 0 && xx % 2 == 0) { c = 0; }
            elif (yy % 2 == 1 && xx % 2 == 1) { c = 2; }
            if (c == wantC) {
                acc = acc + d[yy * w + xx];
                cnt = cnt + 1;
            }
        }
    }
    if (cnt == 0) { return 0.0; }
    return acc / cnt;
}

# Mosaic -> RGB by neighbourhood averaging per color plane.
fn demosaic_bilinear(mos) {
    int b = 0; int h = 0; int out = 0; int w = 0; int x = 0; int y = 0;
    w = mos[0];
    h = mos[1];
    out = [];
    for (y = 0; y < h; ++y) {
        for (x = 0; x < w; ++x) {
            b = (y * w + x) * 3;
            out[b] = demosaic_channel_avg(mos, x, y, w, h, 0, 0);
            out[b + 1] = demosaic_channel_avg(mos, x, y, w, h, 0, 1);
            out[b + 2] = demosaic_channel_avg(mos, x, y, w, h, 0, 2);
        }
    }
    return [w, h, 3, out];
}
