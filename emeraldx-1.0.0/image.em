# =============================================================================
# image.em - images for Emerald
#
# An image is img = [w, h, ch, data]: ch is 1 (gray) or 3 (RGB), data a flat
# float array of length w*h*ch, values in [0, 1], row-major, channels
# interleaved. I/O uses ASCII PPM (P3) / PGM (P2), viewable almost anywhere.
#
# Bulk-processing pattern (fast; avoids per-pixel structure rebuilds):
#     d = img[3];                 # one copy out
#     ... mutate d in place ...
#     img = [img[0], img[1], img[2], d];
# =============================================================================

import mathx;
import arrayx;
import strx;
# emx-scope-safe

fn new_image(w, h, ch, fill) {
    if (fill == nil) { fill = 0.0; }
    return [w, h, ch, arrayx.full(w * h * ch, fill)];
}

fn width(img) { return img[0]; }
fn height(img) { return img[1]; }
fn channels(img) { return img[2]; }

fn idx_of(img, x, y, c) {
    return (y * img[0] + x) * img[2] + c;
}

fn get_px(img, x, y, c) {
    int d = 0;
    d = img[3];
    return d[(y * img[0] + x) * img[2] + c];
}

# Returns a new image (value semantics). For bulk writes see header note.
fn set_px(img, x, y, c, v) {
    int d = 0;
    d = img[3];
    d[(y * img[0] + x) * img[2] + c] = v;
    return [img[0], img[1], img[2], d];
}

# Clamped pixel read (border replication).
fn get_px_clamped(img, x, y, c) {
    int d = 0; int xx = 0; int yy = 0;
    xx = mathx.clampi(x, 0, img[0] - 1);
    yy = mathx.clampi(y, 0, img[1] - 1);
    d = img[3];
    return d[(yy * img[0] + xx) * img[2] + c];
}

# Bilinear sample at float coords (x, y in pixel units).
fn sample_bilinear(img, x, y, c) {
    int bot = 0; int fx = 0; int fy = 0; int top2 = 0; int v00 = 0; int v01 = 0; int v10 = 0; int v11 = 0; int x0 = 0; int y0 = 0;
    x0 = int(floor(x));
    y0 = int(floor(y));
    fx = x - x0;
    fy = y - y0;
    v00 = get_px_clamped(img, x0, y0, c);
    v10 = get_px_clamped(img, x0 + 1, y0, c);
    v01 = get_px_clamped(img, x0, y0 + 1, c);
    v11 = get_px_clamped(img, x0 + 1, y0 + 1, c);
    top2 = v00 * (1.0 - fx) + v10 * fx;
    bot = v01 * (1.0 - fx) + v11 * fx;
    return top2 * (1.0 - fy) + bot * fy;
}

fn clamp01(v) {
    if (v < 0.0) { return 0.0; }
    if (v > 1.0) { return 1.0; }
    return v;
}

# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

fn to_gray(img) {
    int b = 0; int d = 0; int h = 0; int i = 0; int out = 0; int w = 0;
    if (img[2] == 1) { return img; }
    w = img[0];
    h = img[1];
    d = img[3];
    out = [];
    for (i = 0; i < w * h; ++i) {
        b = i * 3;
        out[i] = 0.2126 * d[b] + 0.7152 * d[b + 1] + 0.0722 * d[b + 2];
    }
    return [w, h, 1, out];
}

fn to_rgb(img) {
    int d = 0; int h = 0; int i = 0; int out = 0; int w = 0;
    if (img[2] == 3) { return img; }
    w = img[0];
    h = img[1];
    d = img[3];
    out = [];
    for (i = 0; i < w * h; ++i) {
        out[i * 3] = d[i];
        out[i * 3 + 1] = d[i];
        out[i * 3 + 2] = d[i];
    }
    return [w, h, 3, out];
}

fn get_channel(img, c) {
    int ch = 0; int d = 0; int h = 0; int i = 0; int out = 0; int w = 0;
    w = img[0];
    h = img[1];
    ch = img[2];
    d = img[3];
    out = [];
    for (i = 0; i < w * h; ++i) { out[i] = d[i * ch + c]; }
    return [w, h, 1, out];
}

fn merge_rgb(r, g, b) {
    int bd = 0; int gd = 0; int h = 0; int i = 0; int out = 0; int rd = 0; int w = 0;
    w = r[0];
    h = r[1];
    rd = r[3];
    gd = g[3];
    bd = b[3];
    out = [];
    for (i = 0; i < w * h; ++i) {
        out[i * 3] = rd[i];
        out[i * 3 + 1] = gd[i];
        out[i * 3 + 2] = bd[i];
    }
    return [w, h, 3, out];
}

# Apply a scalar function to every value.
fn apply(img, f) {
    int d = 0; int i = 0; int out = 0;
    d = img[3];
    out = [];
    for (i = 0; i < len(d); ++i) { out[i] = f(d[i]); }
    return [img[0], img[1], img[2], out];
}

fn clamp_image(img) {
    int d = 0; int i = 0; int out = 0;
    d = img[3];
    out = [];
    for (i = 0; i < len(d); ++i) { out[i] = clamp01(d[i]); }
    return [img[0], img[1], img[2], out];
}

fn scale_values(img, s, offset) {
    int d = 0; int i = 0; int out = 0;
    if (offset == nil) { offset = 0.0; }
    d = img[3];
    out = [];
    for (i = 0; i < len(d); ++i) { out[i] = d[i] * s + offset; }
    return [img[0], img[1], img[2], out];
}

# Per-value blend: out = a*(1-t) + b*t.
fn blend(a, b, t) {
    int da = 0; int db = 0; int i = 0; int out = 0;
    da = a[3];
    db = b[3];
    out = [];
    for (i = 0; i < len(da); ++i) { out[i] = da[i] * (1.0 - t) + db[i] * t; }
    return [a[0], a[1], a[2], out];
}

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

fn crop(img, x0, y0, cw, chh) {
    int ch = 0; int d = 0; int dstBase = 0; int k = 0; int out = 0; int srcBase = 0; int w = 0; int y = 0;
    w = img[0];
    ch = img[2];
    d = img[3];
    out = [];
    for (y = 0; y < chh; ++y) {
        srcBase = ((y0 + y) * w + x0) * ch;
        dstBase = y * cw * ch;
        for (k = 0; k < cw * ch; ++k) { out[dstBase + k] = d[srcBase + k]; }
    }
    return [cw, chh, ch, out];
}

fn flip_h(img) {
    int c = 0; int ch = 0; int d = 0; int db = 0; int h = 0; int out = 0; int sb = 0; int w = 0; int x = 0; int y = 0;
    w = img[0];
    h = img[1];
    ch = img[2];
    d = img[3];
    out = [];
    for (y = 0; y < h; ++y) {
        for (x = 0; x < w; ++x) {
            sb = (y * w + (w - 1 - x)) * ch;
            db = (y * w + x) * ch;
            for (c = 0; c < ch; ++c) { out[db + c] = d[sb + c]; }
        }
    }
    return [w, h, ch, out];
}

fn flip_v(img) {
    int c = 0; int ch = 0; int d = 0; int db = 0; int h = 0; int out = 0; int sb = 0; int w = 0; int x = 0; int y = 0;
    w = img[0];
    h = img[1];
    ch = img[2];
    d = img[3];
    out = [];
    for (y = 0; y < h; ++y) {
        for (x = 0; x < w; ++x) {
            sb = ((h - 1 - y) * w + x) * ch;
            db = (y * w + x) * ch;
            for (c = 0; c < ch; ++c) { out[db + c] = d[sb + c]; }
        }
    }
    return [w, h, ch, out];
}

# Bilinear resize to (nw, nh).
fn resize(img, nw, nh) {
    int base = 0; int c = 0; int ch = 0; int h = 0; int out = 0; int srcX = 0; int srcY = 0; int sx = 0; int sy = 0; int w = 0; int x = 0; int y = 0;
    w = img[0];
    h = img[1];
    ch = img[2];
    out = [];
    sx = w * 1.0 / nw;
    sy = h * 1.0 / nh;
    for (y = 0; y < nh; ++y) {
        srcY = (y + 0.5) * sy - 0.5;
        for (x = 0; x < nw; ++x) {
            srcX = (x + 0.5) * sx - 0.5;
            base = (y * nw + x) * ch;
            for (c = 0; c < ch; ++c) {
                out[base + c] = sample_bilinear(img, srcX, srcY, c);
            }
        }
    }
    return [nw, nh, ch, out];
}

# Nearest-neighbour upscale by integer factor (crisp for demos).
fn upscale_nearest(img, factor) {
    int c = 0; int ch = 0; int d = 0; int db = 0; int h = 0; int nh = 0; int nw = 0; int out = 0; int sb = 0; int sy = 0; int w = 0; int x = 0; int y = 0;
    w = img[0];
    h = img[1];
    ch = img[2];
    d = img[3];
    nw = w * factor;
    nh = h * factor;
    out = [];
    for (y = 0; y < nh; ++y) {
        sy = y // factor;
        for (x = 0; x < nw; ++x) {
            sb = (sy * w + x // factor) * ch;
            db = (y * nw + x) * ch;
            for (c = 0; c < ch; ++c) { out[db + c] = d[sb + c]; }
        }
    }
    return [nw, nh, ch, out];
}

# ---------------------------------------------------------------------------
# Drawing (returns new images; batch several shapes via the data pattern)
# ---------------------------------------------------------------------------

fn fill_rect(img, x0, y0, rw, rh, color) {
    int b = 0; int c = 0; int ch = 0; int d = 0; int h = 0; int w = 0; int x = 0; int x1 = 0; int y = 0; int y1 = 0;
    w = img[0];
    h = img[1];
    ch = img[2];
    d = img[3];
    x1 = mathx.mini(x0 + rw, w);
    y1 = mathx.mini(y0 + rh, h);
    for (y = mathx.maxi(y0, 0); y < y1; ++y) {
        for (x = mathx.maxi(x0, 0); x < x1; ++x) {
            b = (y * w + x) * ch;
            for (c = 0; c < ch; ++c) { d[b + c] = color[c]; }
        }
    }
    return [w, h, ch, d];
}

fn draw_line(img, x0, y0, x1, y1, color) {
    int b = 0; int c = 0; int ch = 0; int d = 0; int dx = 0; int dy = 0; int e2 = 0; int err = 0; int h = 0; int sx = 0; int sy = 0; int w = 0; int x = 0; int y = 0;
    w = img[0];
    h = img[1];
    ch = img[2];
    d = img[3];
    dx = abs(x1 - x0);
    dy = 0 - abs(y1 - y0);
    sx = -1;
    if (x0 < x1) { sx = 1; }
    sy = -1;
    if (y0 < y1) { sy = 1; }
    err = dx + dy;
    x = x0;
    y = y0;
    while (true) {
        if (x >= 0 && x < w && y >= 0 && y < h) {
            b = (y * w + x) * ch;
            for (c = 0; c < ch; ++c) { d[b + c] = color[c]; }
        }
        if (x == x1 && y == y1) { break; }
        e2 = 2 * err;
        if (e2 >= dy) { err = err + dy; x = x + sx; }
        if (e2 <= dx) { err = err + dx; y = y + sy; }
    }
    return [w, h, ch, d];
}

fn draw_circle(img, cx, cy, radius, color) {
    int b = 0; int c = 0; int ch = 0; int d = 0; int ddx = 0; int ddy = 0; int h = 0; int r2 = 0; int w = 0; int x = 0; int y = 0;
    w = img[0];
    h = img[1];
    ch = img[2];
    d = img[3];
    r2 = radius * radius;
    for (y = cy - radius; y <= cy + radius; ++y) {
        if (y < 0 || y >= h) { continue; }
        for (x = cx - radius; x <= cx + radius; ++x) {
            if (x < 0 || x >= w) { continue; }
            ddx = x - cx;
            ddy = y - cy;
            if (ddx * ddx + ddy * ddy <= r2) {
                b = (y * w + x) * ch;
                for (c = 0; c < ch; ++c) { d[b + c] = color[c]; }
            }
        }
    }
    return [w, h, ch, d];
}

# ---------------------------------------------------------------------------
# ASCII PPM / PGM I/O
# ---------------------------------------------------------------------------

fn save(img, path) {
    int w = 0; int h = 0; int ch = 0; int d = 0; int i = 0; int v = 0;
    int line = 0; int count = 0; int header = 0;
    w = img[0];
    h = img[1];
    ch = img[2];
    d = img[3];
    File fh = path;
    header = "P3\n";
    if (ch == 1) { header = "P2\n"; }
    fh.write(header + "{w} {h}\n255\n");
    line = "";
    count = 0;
    for (i = 0; i < len(d); ++i) {
        v = int(floor(clamp01(d[i]) * 255.0 + 0.5));
        line = line + string(v) + " ";
        count = count + 1;
        if (count >= 60) {
            fh.append(line + "\n");
            line = "";
            count = 0;
        }
    }
    if (len(line) > 0) { fh.append(line + "\n"); }
    return true;
}

# Load an ASCII PPM (P3) or PGM (P2). Returns an image or nil.
fn load(path) {
    int ch = 0; int cur = 0; int d = 0; int h = 0; int i = 0; int inComment = 0; int magic = 0; int maxv = 0; int need = 0; int text = 0; int toks = 0; int v = 0; int w = 0;
    File fh = path;
    text = fh.read();
    if (text == nil || len(text) == 0) { return nil; }
    toks = [];
    cur = "";
    inComment = false;
    for chv in text {
        if (chv == '#') { inComment = true; }
        if (chv == '\n') { inComment = false; }
        if (inComment) { continue; }
        if (chv == ' ' || chv == '\n' || chv == '\t' || chv == '\r') {
            if (len(cur) > 0) {
                toks[len(toks)] = cur;
                cur = "";
            }
        } else {
            cur = cur + string(chv);
        }
    }
    if (len(cur) > 0) { toks[len(toks)] = cur; }
    if (len(toks) < 4) { return nil; }
    magic = toks[0];
    ch = 0;
    if (magic == "P2") { ch = 1; }
    elif (magic == "P3") { ch = 3; }
    else { return nil; }
    w = strx.parse_int(toks[1]);
    h = strx.parse_int(toks[2]);
    maxv = strx.parse_int(toks[3]);
    if (maxv == nil || maxv <= 0) { maxv = 255; }
    d = [];
    need = w * h * ch;
    for (i = 0; i < need; ++i) {
        v = strx.parse_int(toks[4 + i]);
        if (v == nil) { v = 0; }
        d[i] = v * 1.0 / maxv;
    }
    return [w, h, ch, d];
}

# Print a tiny gray image as ASCII art (for quick terminal checks).
fn print_ascii(img, chars) {
    int d = 0; int g = 0; int h = 0; int k = 0; int line = 0; int nlev = 0; int v = 0; int w = 0; int x = 0; int y = 0;
    if (chars == nil) { chars = " .:-=+*#%@"; }
    g = to_gray(img);
    w = g[0];
    h = g[1];
    d = g[3];
    nlev = len(chars);
    for (y = 0; y < h; ++y) {
        line = "";
        for (x = 0; x < w; ++x) {
            v = clamp01(d[y * w + x]);
            k = int(floor(v * (nlev - 1) + 0.5));
            line = line + string(chars[k]);
        }
        print(line);
    }
}
