# =============================================================================
# texture.em - procedural textures and texture sampling
#
#   checker, stripes, gradient_tex                    patterns
#   value_noise, fbm, turbulence, marble, wood        noise-based
#   normal_from_height                                bump -> normal map
#   sample_uv                                         bilinear UV lookup
#
# All generators return images ([w, h, ch, data]). UV space is [0,1]^2 with
# (0,0) at the top-left; sample_uv wraps (tiles).
# =============================================================================

import mathx;
import arrayx;
import prng;
import image;
# emx-scope-safe

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

# Bilinear sample with wrap-around UVs.
fn sample_uv(img, u, v, c) {
    int h = 0; int uu = 0; int vv = 0; int w = 0;
    w = img[0];
    h = img[1];
    uu = u - floor(u);
    vv = v - floor(v);
    return image.sample_bilinear(img, uu * (w - 1), vv * (h - 1), c);
}

# RGB sample helper -> [r, g, b].
fn sample_uv_rgb(img, u, v) {
    int g = 0;
    if (img[2] == 1) {
        g = sample_uv(img, u, v, 0);
        return [g, g, g];
    }
    return [sample_uv(img, u, v, 0), sample_uv(img, u, v, 1),
            sample_uv(img, u, v, 2)];
}

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

fn checker(w, h, cells, colorA, colorB) {
    int b = 0; int col = 0; int cx = 0; int cy = 0; int out = 0; int x = 0; int y = 0;
    if (colorA == nil) { colorA = [0.9, 0.9, 0.9]; }
    if (colorB == nil) { colorB = [0.15, 0.15, 0.15]; }
    out = [];
    for (y = 0; y < h; ++y) {
        cy = (y * cells) // h;
        for (x = 0; x < w; ++x) {
            cx = (x * cells) // w;
            col = colorA;
            if ((cx + cy) % 2 == 1) { col = colorB; }
            b = (y * w + x) * 3;
            out[b] = col[0];
            out[b + 1] = col[1];
            out[b + 2] = col[2];
        }
    }
    return [w, h, 3, out];
}

fn stripes(w, h, count, horizontal, colorA, colorB) {
    int b = 0; int col = 0; int k = 0; int out = 0; int x = 0; int y = 0;
    if (colorA == nil) { colorA = [0.85, 0.3, 0.2]; }
    if (colorB == nil) { colorB = [0.95, 0.9, 0.8]; }
    out = [];
    for (y = 0; y < h; ++y) {
        for (x = 0; x < w; ++x) {
            k = (x * count) // w;
            if (horizontal == true) { k = (y * count) // h; }
            col = colorA;
            if (k % 2 == 1) { col = colorB; }
            b = (y * w + x) * 3;
            out[b] = col[0];
            out[b + 1] = col[1];
            out[b + 2] = col[2];
        }
    }
    return [w, h, 3, out];
}

# Linear vertical gradient between two RGB colors.
fn gradient_tex(w, h, top2, bottom) {
    int b = 0; int c = 0; int out = 0; int t = 0; int x = 0; int y = 0;
    out = [];
    for (y = 0; y < h; ++y) {
        t = y * 1.0 / (h - 1);
        for (x = 0; x < w; ++x) {
            b = (y * w + x) * 3;
            for (c = 0; c < 3; ++c) {
                out[b + c] = top2[c] * (1.0 - t) + bottom[c] * t;
            }
        }
    }
    return [w, h, 3, out];
}

# ---------------------------------------------------------------------------
# Value noise: a lattice of deterministic pseudo-random values, smoothly
# interpolated. hash2 gives a repeatable value per lattice point - no state.
# ---------------------------------------------------------------------------

fn hash2(ix, iy, seed) {
    int n = 0;
    # integer scramble (all intermediates < 2^53)
    n = ix * 374761 + iy * 668265 + seed * 1274126;
    n = (n * 1103515245 + 12345) % 2147483648;
    n = (n * 1103515245 + 12345) % 2147483648;
    return n / 2147483648.0;
}

fn smoothstep(t) { return t * t * (3.0 - 2.0 * t); }

# Single octave of value noise at (x, y); frequency in lattice cells.
fn value_noise_at(x, y, seed) {
    int a = 0; int b = 0; int fx = 0; int fy = 0; int ix = 0; int iy = 0; int sx = 0; int sy = 0; int v00 = 0; int v01 = 0; int v10 = 0; int v11 = 0;
    ix = int(floor(x));
    iy = int(floor(y));
    fx = x - ix;
    fy = y - iy;
    v00 = hash2(ix, iy, seed);
    v10 = hash2(ix + 1, iy, seed);
    v01 = hash2(ix, iy + 1, seed);
    v11 = hash2(ix + 1, iy + 1, seed);
    sx = smoothstep(fx);
    sy = smoothstep(fy);
    a = v00 * (1.0 - sx) + v10 * sx;
    b = v01 * (1.0 - sx) + v11 * sx;
    return a * (1.0 - sy) + b * sy;
}

# Fractal Brownian motion: octaves of value noise. Returns value in ~[0,1].
fn fbm_at(x, y, octaves, seed) {
    int amp = 0; int freq = 0; int norm = 0; int o = 0; int total = 0;
    total = 0.0;
    amp = 0.5;
    freq = 1.0;
    norm = 0.0;
    for (o = 0; o < octaves; ++o) {
        total = total + amp * value_noise_at(x * freq, y * freq, seed + o * 131);
        norm = norm + amp;
        amp = amp * 0.5;
        freq = freq * 2.0;
    }
    return total / norm;
}

# Gray noise image; scale = lattice cells across the width.
fn value_noise(w, h, scale, octaves, seed) {
    int out = 0; int x = 0; int y = 0;
    if (octaves == nil) { octaves = 4; }
    out = [];
    for (y = 0; y < h; ++y) {
        for (x = 0; x < w; ++x) {
            out[y * w + x] = fbm_at(x * scale / w, y * scale / w, octaves, seed);
        }
    }
    return [w, h, 1, out];
}

# Absolute-value fbm ("turbulence") - billowy look.
fn turbulence(w, h, scale, octaves, seed) {
    int out = 0; int v = 0; int x = 0; int y = 0;
    if (octaves == nil) { octaves = 4; }
    out = [];
    for (y = 0; y < h; ++y) {
        for (x = 0; x < w; ++x) {
            v = fbm_at(x * scale / w, y * scale / w, octaves, seed);
            out[y * w + x] = abs(2.0 * v - 1.0);
        }
    }
    return [w, h, 1, out];
}

# Marble: sine bands warped by turbulence.
fn marble(w, h, seed) {
    int base = 0; int bd = 0; int i = 0; int out = 0; int v = 0; int x = 0; int y = 0;
    base = turbulence(w, h, 4.0, 4, seed);
    bd = base[3];
    out = [];
    for (y = 0; y < h; ++y) {
        for (x = 0; x < w; ++x) {
            i = y * w + x;
            v = sin((x * 8.0 / w + bd[i] * 6.0) * pi);
            out[i] = 0.5 + 0.5 * v;
        }
    }
    return [w, h, 1, out];
}

# Wood rings: radial distance warped by noise.
fn wood(w, h, rings, seed) {
    int base = 0; int bd = 0; int cx = 0; int cy = 0; int dx = 0; int dy = 0; int i = 0; int out = 0; int r = 0; int v = 0; int x = 0; int y = 0;
    if (rings == nil) { rings = 6.0; }
    base = value_noise(w, h, 3.0, 3, seed);
    bd = base[3];
    out = [];
    cx = w / 2.0;
    cy = h / 2.0;
    for (y = 0; y < h; ++y) {
        for (x = 0; x < w; ++x) {
            i = y * w + x;
            dx = (x - cx) / w;
            dy = (y - cy) / h;
            r = sqrt(dx * dx + dy * dy) + bd[i] * 0.12;
            v = r * rings;
            out[i] = 0.55 + 0.45 * sin(v * mathx.TAU);
        }
    }
    return [w, h, 1, out];
}

# ---------------------------------------------------------------------------
# Height map -> tangent-space normal map (RGB encodes xyz*0.5+0.5).
# ---------------------------------------------------------------------------

fn normal_from_height(heightImg, strength) {
    int b = 0; int h = 0; int hd = 0; int hl = 0; int hr = 0; int hu = 0; int invLen = 0; int nx = 0; int ny = 0; int nz = 0; int out = 0; int w = 0; int x = 0; int y = 0;
    if (strength == nil) { strength = 2.0; }
    w = heightImg[0];
    h = heightImg[1];
    out = [];
    for (y = 0; y < h; ++y) {
        for (x = 0; x < w; ++x) {
            hl = image.get_px_clamped(heightImg, x - 1, y, 0);
            hr = image.get_px_clamped(heightImg, x + 1, y, 0);
            hu = image.get_px_clamped(heightImg, x, y - 1, 0);
            hd = image.get_px_clamped(heightImg, x, y + 1, 0);
            nx = (hl - hr) * strength;
            ny = (hu - hd) * strength;
            nz = 1.0;
            invLen = 1.0 / sqrt(nx * nx + ny * ny + nz * nz);
            b = (y * w + x) * 3;
            out[b] = nx * invLen * 0.5 + 0.5;
            out[b + 1] = ny * invLen * 0.5 + 0.5;
            out[b + 2] = nz * invLen * 0.5 + 0.5;
        }
    }
    return [w, h, 3, out];
}
