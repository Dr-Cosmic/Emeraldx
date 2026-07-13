# =============================================================================
# imgproc.em - classic image processing on emeraldx images
#
#   convolve2d, gaussian_kernel, box_kernel, sharpen_kernel, laplacian_kernel
#   sobel (gradient magnitude + direction), canny (full pipeline)
#   threshold, otsu_threshold, invert, equalize_hist
#   erode / dilate / opening / closing (binary morphology)
#   connected_components (explicit stack flood fill)
#   harris_corners, match_template_ncc
#
# All functions take and return images ([w, h, ch, data]); most expect
# single-channel input - call image.to_gray first.
# =============================================================================

import mathx;
import arrayx;
import image;
# emx-scope-safe

# ---------------------------------------------------------------------------
# Convolution
# ---------------------------------------------------------------------------

# kernel = [kh, kw, flat weights]; border replicated; single channel.
fn convolve2d(img, kernel) {
    int d = 0; int h = 0; int kBase = 0; int kd = 0; int kh = 0; int kw = 0; int kx = 0; int ky = 0; int out = 0; int rowBase = 0; int rx = 0; int ry = 0; int s = 0; int w = 0; int x = 0; int xx = 0; int y = 0; int yy = 0;
    w = img[0];
    h = img[1];
    d = img[3];
    kh = kernel[0];
    kw = kernel[1];
    kd = kernel[2];
    ry = kh // 2;
    rx = kw // 2;
    out = [];
    for (y = 0; y < h; ++y) {
        for (x = 0; x < w; ++x) {
            s = 0.0;
            for (ky = 0; ky < kh; ++ky) {
                yy = mathx.clampi(y + ky - ry, 0, h - 1);
                rowBase = yy * w;
                kBase = ky * kw;
                for (kx = 0; kx < kw; ++kx) {
                    xx = mathx.clampi(x + kx - rx, 0, w - 1);
                    s = s + d[rowBase + xx] * kd[kBase + kx];
                }
            }
            out[y * w + x] = s;
        }
    }
    return [w, h, 1, out];
}

fn box_kernel(size) {
    int kd = 0; int n = 0;
    n = size * size;
    kd = arrayx.full(n, 1.0 / n);
    return [size, size, kd];
}

fn gaussian_kernel(size, sigma) {
    int dx = 0; int dy = 0; int i = 0; int kd = 0; int r = 0; int s = 0; int v = 0; int x = 0; int y = 0;
    if (sigma == nil) { sigma = size / 3.0; }
    r = size // 2;
    kd = [];
    s = 0.0;
    for (y = 0; y < size; ++y) {
        for (x = 0; x < size; ++x) {
            dx = x - r;
            dy = y - r;
            v = exp(0.0 - (dx * dx + dy * dy) / (2.0 * sigma * sigma));
            kd[y * size + x] = v;
            s = s + v;
        }
    }
    for (i = 0; i < len(kd); ++i) { kd[i] = kd[i] / s; }
    return [size, size, kd];
}

fn sharpen_kernel() {
    return [3, 3, [0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0]];
}

fn laplacian_kernel() {
    return [3, 3, [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0]];
}

fn emboss_kernel() {
    return [3, 3, [-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0]];
}

fn gaussian_blur(img, size, sigma) {
    return convolve2d(img, gaussian_kernel(size, sigma));
}

# ---------------------------------------------------------------------------
# Gradients / edges
# ---------------------------------------------------------------------------

# Sobel gradients. Returns [magnitude, direction] images; direction in
# radians (-pi..pi), magnitude unnormalized.
fn sobel(img) {
    int ang = 0; int gx = 0; int gxd = 0; int gy = 0; int gyd = 0; int h = 0; int i = 0; int kx = 0; int ky = 0; int mag = 0; int w = 0;
    kx = [3, 3, [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0]];
    ky = [3, 3, [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0]];
    gx = convolve2d(img, kx);
    gy = convolve2d(img, ky);
    w = img[0];
    h = img[1];
    gxd = gx[3];
    gyd = gy[3];
    mag = [];
    ang = [];
    for (i = 0; i < w * h; ++i) {
        mag[i] = sqrt(gxd[i] * gxd[i] + gyd[i] * gyd[i]);
        ang[i] = mathx.atan2(gyd[i], gxd[i]);
    }
    return [[w, h, 1, mag], [w, h, 1, ang]];
}

# Canny edge detector: gaussian blur -> sobel -> non-max suppression ->
# double threshold -> hysteresis (explicit stack). Returns a binary image.
fn canny(img, lowT, highT) {
    int a = 0; int ang = 0; int angImg = 0; int blurred = 0; int deg = 0; int dx = 0; int dy = 0; int h = 0; int i = 0; int j = 0; int mag = 0; int magImg = 0; int mx = 0; int nms = 0; int out = 0; int popped = 0; int q0 = 0; int q1 = 0; int sb = 0; int stack = 0; int w = 0; int x = 0; int xx = 0; int y = 0; int yy = 0;
    if (lowT == nil) { lowT = 0.1; }
    if (highT == nil) { highT = 0.25; }
    w = img[0];
    h = img[1];
    blurred = gaussian_blur(img, 5, 1.2);
    sb = sobel(blurred);
    magImg = sb[0];
    angImg = sb[1];
    mag = magImg[3];
    ang = angImg[3];
    # normalize magnitude to [0,1]
    mx = arrayx.maxv(mag);
    if (mx < 1.0e-12) { mx = 1.0; }
    for (i = 0; i < len(mag); ++i) { mag[i] = mag[i] / mx; }
    # non-maximum suppression along the gradient direction
    nms = arrayx.zerosf(w * h);
    for (y = 1; y < h - 1; ++y) {
        for (x = 1; x < w - 1; ++x) {
            i = y * w + x;
            a = ang[i];
            # quantize direction to 0, 45, 90, 135 degrees
            deg = a * 180.0 / pi;
            if (deg < 0.0) { deg = deg + 180.0; }
            q0 = 0; q1 = 0;
            if (deg < 22.5 || deg >= 157.5) {
                q0 = i - 1; q1 = i + 1;                       # horizontal
            } elif (deg < 67.5) {
                q0 = i - w + 1; q1 = i + w - 1;               # 45
            } elif (deg < 112.5) {
                q0 = i - w; q1 = i + w;                       # vertical
            } else {
                q0 = i - w - 1; q1 = i + w + 1;               # 135
            }
            if (mag[i] >= mag[q0] && mag[i] >= mag[q1]) { nms[i] = mag[i]; }
        }
    }
    # double threshold + hysteresis
    out = arrayx.zerosf(w * h);
    stack = [];
    for (i = 0; i < w * h; ++i) {
        if (nms[i] >= highT) {
            out[i] = 1.0;
            stack[len(stack)] = i;
        }
    }
    while (len(stack) > 0) {
        popped = arrayx.pop(stack);
        stack = popped[0];
        i = popped[1];
        y = i // w;
        x = i % w;
        for (dy = -1; dy <= 1; ++dy) {
            for (dx = -1; dx <= 1; ++dx) {
                yy = y + dy;
                xx = x + dx;
                if (yy < 0 || yy >= h || xx < 0 || xx >= w) { continue; }
                j = yy * w + xx;
                if (out[j] == 0.0 && nms[j] >= lowT) {
                    out[j] = 1.0;
                    stack[len(stack)] = j;
                }
            }
        }
    }
    return [w, h, 1, out];
}

# ---------------------------------------------------------------------------
# Thresholding / histogram
# ---------------------------------------------------------------------------

fn threshold(img, t) {
    int d = 0; int i = 0; int out = 0;
    d = img[3];
    out = [];
    for (i = 0; i < len(d); ++i) {
        if (d[i] >= t) { out[i] = 1.0; } else { out[i] = 0.0; }
    }
    return [img[0], img[1], img[2], out];
}

fn invert(img) {
    int d = 0; int i = 0; int out = 0;
    d = img[3];
    out = [];
    for (i = 0; i < len(d); ++i) { out[i] = 1.0 - d[i]; }
    return [img[0], img[1], img[2], out];
}

# Otsu's method on a 256-bin histogram of a gray image. Returns the threshold.
fn otsu_threshold(img) {
    int b = 0; int best = 0; int bestT = 0; int between = 0; int d = 0; int hist = 0; int i = 0; int mB = 0; int mF = 0; int n = 0; int sumAll = 0; int sumB = 0; int total = 0; int wB = 0; int wF = 0;
    d = img[3];
    n = len(d);
    hist = arrayx.zerosf(256);
    for (i = 0; i < n; ++i) {
        b = int(floor(image.clamp01(d[i]) * 255.0));
        hist[b] = hist[b] + 1;
    }
    total = n * 1.0;
    sumAll = 0.0;
    for (b = 0; b < 256; ++b) { sumAll = sumAll + b * hist[b]; }
    sumB = 0.0;
    wB = 0.0;
    best = 0.0;
    bestT = 127;
    for (b = 0; b < 256; ++b) {
        wB = wB + hist[b];
        if (wB == 0.0) { continue; }
        wF = total - wB;
        if (wF == 0.0) { break; }
        sumB = sumB + b * hist[b];
        mB = sumB / wB;
        mF = (sumAll - sumB) / wF;
        between = wB * wF * (mB - mF) * (mB - mF);
        if (between > best) { best = between; bestT = b; }
    }
    return (bestT + 1.0) / 255.0;
}

# Histogram equalization of a gray image.
fn equalize_hist(img) {
    int acc = 0; int b = 0; int cdf = 0; int cdfMin = 0; int d = 0; int denom = 0; int h = 0; int hist = 0; int i = 0; int n = 0; int out = 0; int w = 0;
    w = img[0];
    h = img[1];
    d = img[3];
    n = len(d);
    hist = arrayx.zerosf(256);
    for (i = 0; i < n; ++i) {
        b = int(floor(image.clamp01(d[i]) * 255.0));
        hist[b] = hist[b] + 1;
    }
    cdf = arrayx.zerosf(256);
    acc = 0.0;
    for (b = 0; b < 256; ++b) {
        acc = acc + hist[b];
        cdf[b] = acc;
    }
    cdfMin = 0.0;
    for (b = 0; b < 256; ++b) {
        if (cdf[b] > 0.0) { cdfMin = cdf[b]; break; }
    }
    denom = n - cdfMin;
    if (denom < 1.0) { denom = 1.0; }
    out = [];
    for (i = 0; i < n; ++i) {
        b = int(floor(image.clamp01(d[i]) * 255.0));
        out[i] = (cdf[b] - cdfMin) / denom;
    }
    return [w, h, 1, out];
}

# ---------------------------------------------------------------------------
# Binary morphology (input: 0/1 gray image)
# ---------------------------------------------------------------------------

fn erode(img, radius) {
    int d = 0; int dx = 0; int dy = 0; int h = 0; int keep = 0; int out = 0; int w = 0; int x = 0; int xx = 0; int y = 0; int yy = 0;
    if (radius == nil) { radius = 1; }
    w = img[0];
    h = img[1];
    d = img[3];
    out = [];
    for (y = 0; y < h; ++y) {
        for (x = 0; x < w; ++x) {
            keep = 1.0;
            for (dy = 0 - radius; dy <= radius; ++dy) {
                for (dx = 0 - radius; dx <= radius; ++dx) {
                    yy = mathx.clampi(y + dy, 0, h - 1);
                    xx = mathx.clampi(x + dx, 0, w - 1);
                    if (d[yy * w + xx] < 0.5) { keep = 0.0; }
                }
            }
            out[y * w + x] = keep;
        }
    }
    return [w, h, 1, out];
}

fn dilate(img, radius) {
    int d = 0; int dx = 0; int dy = 0; int h = 0; int hit = 0; int out = 0; int w = 0; int x = 0; int xx = 0; int y = 0; int yy = 0;
    if (radius == nil) { radius = 1; }
    w = img[0];
    h = img[1];
    d = img[3];
    out = [];
    for (y = 0; y < h; ++y) {
        for (x = 0; x < w; ++x) {
            hit = 0.0;
            for (dy = 0 - radius; dy <= radius; ++dy) {
                for (dx = 0 - radius; dx <= radius; ++dx) {
                    yy = mathx.clampi(y + dy, 0, h - 1);
                    xx = mathx.clampi(x + dx, 0, w - 1);
                    if (d[yy * w + xx] >= 0.5) { hit = 1.0; }
                }
            }
            out[y * w + x] = hit;
        }
    }
    return [w, h, 1, out];
}

fn opening(img, radius) { return dilate(erode(img, radius), radius); }
fn closing(img, radius) { return erode(dilate(img, radius), radius); }

# ---------------------------------------------------------------------------
# Connected components (4-connectivity) on a binary image.
# Returns [labelImage, count]; labels start at 1, background 0.
# ---------------------------------------------------------------------------

fn connected_components(img) {
    int d = 0; int h = 0; int i = 0; int j = 0; int labels = 0; int next = 0; int popped = 0; int stack = 0; int start = 0; int w = 0; int x = 0; int y = 0;
    w = img[0];
    h = img[1];
    d = img[3];
    labels = arrayx.zerosf(w * h);
    next = 1;
    for (start = 0; start < w * h; ++start) {
        if (d[start] < 0.5 || labels[start] != 0.0) { continue; }
        stack = [];
        stack[0] = start;
        labels[start] = next;
        while (len(stack) > 0) {
            popped = arrayx.pop(stack);
            stack = popped[0];
            i = popped[1];
            y = i // w;
            x = i % w;
            if (x > 0) {
                j = i - 1;
                if (d[j] >= 0.5 && labels[j] == 0.0) { labels[j] = next; stack[len(stack)] = j; }
            }
            if (x < w - 1) {
                j = i + 1;
                if (d[j] >= 0.5 && labels[j] == 0.0) { labels[j] = next; stack[len(stack)] = j; }
            }
            if (y > 0) {
                j = i - w;
                if (d[j] >= 0.5 && labels[j] == 0.0) { labels[j] = next; stack[len(stack)] = j; }
            }
            if (y < h - 1) {
                j = i + w;
                if (d[j] >= 0.5 && labels[j] == 0.0) { labels[j] = next; stack[len(stack)] = j; }
            }
        }
        next = next + 1;
    }
    return [[w, h, 1, labels], next - 1];
}

# ---------------------------------------------------------------------------
# Harris corners. Returns array of [x, y, response] sorted by response,
# strongest first, with simple local non-max suppression.
# ---------------------------------------------------------------------------

fn harris_corners(img, k, maxCorners) {
    int a = 0; int b = 0; int c = 0; int cand = 0; int count = 0; int det = 0; int dx = 0; int dy = 0; int g = 0; int gx = 0; int gxd = 0; int gy = 0; int gyd = 0; int h = 0; int i = 0; int isMax = 0; int ixx = 0; int ixy = 0; int iyy = 0; int kx = 0; int ky = 0; int order = 0; int out = 0; int r = 0; int resp = 0; int scores = 0; int sxx = 0; int sxy = 0; int syy = 0; int t = 0; int take = 0; int tr = 0; int w = 0; int x = 0; int y = 0;
    if (k == nil) { k = 0.05; }
    if (maxCorners == nil) { maxCorners = 32; }
    w = img[0];
    h = img[1];
    kx = [3, 3, [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0]];
    ky = [3, 3, [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0]];
    gx = convolve2d(img, kx);
    gy = convolve2d(img, ky);
    gxd = gx[3];
    gyd = gy[3];
    ixx = [];
    iyy = [];
    ixy = [];
    for (i = 0; i < w * h; ++i) {
        ixx[i] = gxd[i] * gxd[i];
        iyy[i] = gyd[i] * gyd[i];
        ixy[i] = gxd[i] * gyd[i];
    }
    g = gaussian_kernel(3, 1.0);
    sxx = convolve2d([w, h, 1, ixx], g);
    syy = convolve2d([w, h, 1, iyy], g);
    sxy = convolve2d([w, h, 1, ixy], g);
    a = sxx[3];
    b = syy[3];
    c = sxy[3];
    resp = [];
    for (i = 0; i < w * h; ++i) {
        det = a[i] * b[i] - c[i] * c[i];
        tr = a[i] + b[i];
        resp[i] = det - k * tr * tr;
    }
    # local 3x3 non-max suppression, then pick the strongest
    cand = [];
    scores = [];
    for (y = 1; y < h - 1; ++y) {
        for (x = 1; x < w - 1; ++x) {
            i = y * w + x;
            r = resp[i];
            if (r <= 0.0) { continue; }
            isMax = true;
            for (dy = -1; dy <= 1; ++dy) {
                for (dx = -1; dx <= 1; ++dx) {
                    if (resp[(y + dy) * w + (x + dx)] > r) { isMax = false; }
                }
            }
            if (isMax) {
                cand[len(cand)] = [x, y, r];
                scores[len(scores)] = r;
            }
        }
    }
    order = arrayx.argsort(scores);
    out = [];
    count = len(cand);
    take = mathx.mini(maxCorners, count);
    for (t = 0; t < take; ++t) {
        out[t] = cand[order[count - 1 - t]];
    }
    return out;
}

# ---------------------------------------------------------------------------
# Normalized cross-correlation template matching.
# Returns [bestX, bestY, bestScore] for the top-left corner of the match.
# ---------------------------------------------------------------------------

fn match_template_ncc(img, tmpl) {
    int base = 0; int bestS = 0; int bestX = 0; int bestY = 0; int d = 0; int den = 0; int dv = 0; int h = 0; int i = 0; int mean2 = 0; int num = 0; int score = 0; int tMean = 0; int tStd = 0; int tVar = 0; int tbase = 0; int td = 0; int th = 0; int tn = 0; int tw = 0; int tx = 0; int ty = 0; int var2 = 0; int w = 0; int x = 0; int y = 0;
    w = img[0];
    h = img[1];
    d = img[3];
    tw = tmpl[0];
    th = tmpl[1];
    td = tmpl[3];
    tn = tw * th;
    tMean = arrayx.sumv(td) / tn;
    tVar = 0.0;
    for (i = 0; i < tn; ++i) {
        dv = td[i] - tMean;
        tVar = tVar + dv * dv;
    }
    tStd = sqrt(tVar);
    if (tStd < 1.0e-12) { tStd = 1.0; }
    bestX = 0;
    bestY = 0;
    bestS = -2.0;
    for (y = 0; y + th <= h; ++y) {
        for (x = 0; x + tw <= w; ++x) {
            mean2 = 0.0;
            for (ty = 0; ty < th; ++ty) {
                base = (y + ty) * w + x;
                for (tx = 0; tx < tw; ++tx) { mean2 = mean2 + d[base + tx]; }
            }
            mean2 = mean2 / tn;
            num = 0.0;
            var2 = 0.0;
            for (ty = 0; ty < th; ++ty) {
                base = (y + ty) * w + x;
                tbase = ty * tw;
                for (tx = 0; tx < tw; ++tx) {
                    dv = d[base + tx] - mean2;
                    num = num + dv * (td[tbase + tx] - tMean);
                    var2 = var2 + dv * dv;
                }
            }
            den = sqrt(var2) * tStd;
            if (den < 1.0e-12) { continue; }
            score = num / den;
            if (score > bestS) {
                bestS = score;
                bestX = x;
                bestY = y;
            }
        }
    }
    return [bestX, bestY, bestS];
}
