# =============================================================================
# nerf.em - a tiny Neural Radiance Field, trained end to end in Emerald
#
# A radiance field maps a 3D point to color + density:
#     field(p) -> [r, g, b, sigma]
# implemented as an MLP (nn.em) over a positional encoding of p. Images are
# formed by EMISSION-ABSORPTION VOLUME RENDERING along camera rays:
#
#     alpha_i = 1 - exp(-sigma_i * delta_i)
#     T_i     = prod_{j<i} (1 - alpha_j)          (transmittance)
#     C       = sum_i T_i * alpha_i * c_i + T_N * background
#
# Training minimizes ||C - C_target||^2. The gradient of the loss flows
# ANALYTICALLY through the compositing (suffix-sum trick below) and then
# through the MLP via nn.backward - no autodiff framework, just calculus.
#
#   field = nerf.make_field(2, 24, seed)        L=2 enc, 24-wide MLP
#   r = nerf.render_ray(field, ro, rd, near, far, n, bg)   -> [rgb, aux]
#   g = nerf.ray_grads(field, aux, dLdC)        -> [gradW, gradB]
#   ... accumulate over rays, nn.apply_grads, repeat.
# =============================================================================

import mathx;
import arrayx;
import vector;
import mat4;
import prng;
import nn;
import image;
# emx-scope-safe

# ---------------------------------------------------------------------------
# Positional encoding: p -> [p, sin(2^l pi p), cos(2^l pi p)] for l < L.
# ---------------------------------------------------------------------------

fn posenc_dim(L) { return 3 + 3 * 2 * L; }

fn posenc(p, L) {
    int c = 0; int f = 0; int l = 0; int out = 0;
    out = [p[0], p[1], p[2]];
    for (l = 0; l < L; ++l) {
        f = (2.0 ** l) * pi;
        for (c = 0; c < 3; ++c) {
            out[len(out)] = sin(f * p[c]);
            out[len(out)] = cos(f * p[c]);
        }
    }
    return out;
}

# ---------------------------------------------------------------------------
# Field = [net, L]. Net maps posenc -> [r_raw, g_raw, b_raw, sigma_raw];
# rgb passes through a sigmoid, sigma through softplus (both smooth,
# differentiable, and positive where they must be).
# ---------------------------------------------------------------------------

# sigmaBias < 0 starts the field nearly transparent, which together with a
# white background is the reliable recipe for fast convergence (empty space
# earns gradient pressure from the very first epoch). Default -2.0.
fn make_field(L, hidden, seed, sigmaBias) {
    int b = 0; int net = 0; int bs = 0; int lastB = 0;
    if (sigmaBias == nil) { sigmaBias = -2.0; }
    b = nn.build([posenc_dim(L), hidden, hidden, 4],
                 ["relu", "relu", "linear"], seed);
    net = b[0];
    bs = net[3];
    lastB = bs[2];
    lastB[3] = sigmaBias;
    bs[2] = lastB;
    return [[[net[0], net[1], net[2], bs], L], b[1]];
}

fn softplus(x) {
    if (x > 20.0) { return x; }
    return ln(1.0 + exp(x));
}

fn softplus_deriv(x) { return mathx.logistic(x); }

# Query the field. Returns [r, g, b, sigma, cache, raw] - cache/raw kept for
# the backward pass.
fn query(field, p) {
    int L = 0; int b = 0; int cache = 0; int enc = 0; int g = 0; int net = 0; int r = 0; int raw = 0; int sigma = 0;
    net = field[0];
    L = field[1];
    enc = posenc(p, L);
    cache = nn.forward(net, enc);
    raw = nn.output(cache);
    r = mathx.logistic(raw[0]);
    g = mathx.logistic(raw[1]);
    b = mathx.logistic(raw[2]);
    sigma = softplus(raw[3]);
    return [r, g, b, sigma, cache, raw];
}

# ---------------------------------------------------------------------------
# Volume rendering along one ray.
# Returns [rgb, aux]; aux stores everything the backward pass needs:
# aux = [caches, raws, colors, sigmas, alphas, Ts, deltas, bg]
# ---------------------------------------------------------------------------

fn render_ray(field, ro, rd, near, far, nSamples, bg) {
    int a = 0; int alphas = 0; int aux = 0; int caches = 0; int cb = 0; int cg = 0; int colors = 0; int cr = 0; int dt = 0; int i = 0; int p = 0; int q = 0; int raws = 0; int sigmas = 0; int t = 0; int trans = 0; int ts = 0; int wgt = 0;
    if (bg == nil) { bg = [0.0, 0.0, 0.0]; }
    dt = (far - near) / nSamples;
    colors = [];
    sigmas = [];
    alphas = [];
    caches = [];
    raws = [];
    trans = 1.0;
    ts = [];
    cr = 0.0; cg = 0.0; cb = 0.0;
    for (i = 0; i < nSamples; ++i) {
        t = near + dt * (i + 0.5);
        p = [ro[0] + rd[0] * t, ro[1] + rd[1] * t, ro[2] + rd[2] * t];
        q = query(field, p);
        colors[i] = [q[0], q[1], q[2]];
        sigmas[i] = q[3];
        caches[i] = q[4];
        raws[i] = q[5];
        a = 1.0 - exp(0.0 - q[3] * dt);
        alphas[i] = a;
        ts[i] = trans;
        wgt = trans * a;
        cr = cr + wgt * q[0];
        cg = cg + wgt * q[1];
        cb = cb + wgt * q[2];
        trans = trans * (1.0 - a);
    }
    cr = cr + trans * bg[0];
    cg = cg + trans * bg[1];
    cb = cb + trans * bg[2];
    aux = [caches, raws, colors, sigmas, alphas, ts, dt, bg, trans];
    return [[cr, cg, cb], aux];
}

# ---------------------------------------------------------------------------
# Backward: given dL/dC (3-vector), push gradients through the compositing
# and the MLP. Suffix-sum trick:
#
#   dL/dc_i     = T_i alpha_i * dLdC                    (per channel)
#   dL/dalpha_i = (dLdC . c_i) T_i  -  S_i / (1-alpha_i)
#       where S_i = sum_{j>i} (dLdC . c_j) T_j alpha_j + (dLdC . bg) T_N
#   dL/dsigma_i = dL/dalpha_i * dt * (1 - alpha_i)
#
# Returns [gradW, gradB] summed over all samples of the ray.
# ---------------------------------------------------------------------------

fn ray_grads(field, aux, dLdC) {
    int a = 0; int acc = 0; int accB = 0; int accW = 0; int alphas = 0; int bg = 0; int c = 0; int caches = 0; int colors = 0; int dAlpha = 0; int dOut = 0; int dSigma = 0; int dot = 0; int dt = 0; int g = 0; int i = 0; int n = 0; int net = 0; int raw = 0; int raws = 0; int sb = 0; int sg = 0; int sr = 0; int suffix = 0; int tI = 0; int tn = 0; int ts = 0; int w = 0; int z = 0;
    net = field[0];
    caches = aux[0];
    raws = aux[1];
    colors = aux[2];
    alphas = aux[4];
    ts = aux[5];
    dt = aux[6];
    bg = aux[7];
    tn = aux[8];
    n = len(colors);
    z = nn.zero_grads(net);
    accW = z[0];
    accB = z[1];
    # suffix sums of downstream weighted-color contributions
    suffix = (dLdC[0] * bg[0] + dLdC[1] * bg[1] + dLdC[2] * bg[2]) * tn;
    for (i = n - 1; i >= 0; --i) {
        c = colors[i];
        a = alphas[i];
        tI = ts[i];
        w = tI * a;
        dot = dLdC[0] * c[0] + dLdC[1] * c[1] + dLdC[2] * c[2];
        dAlpha = dot * tI;
        if (a < 0.999999) {
            dAlpha = dAlpha - suffix / (1.0 - a);
        }
        dSigma = dAlpha * dt * (1.0 - a);
        # raw-output gradient: rgb heads via sigmoid', sigma via softplus'
        raw = raws[i];
        dOut = [];
        sr = mathx.logistic(raw[0]);
        sg = mathx.logistic(raw[1]);
        sb = mathx.logistic(raw[2]);
        dOut[0] = dLdC[0] * w * sr * (1.0 - sr);
        dOut[1] = dLdC[1] * w * sg * (1.0 - sg);
        dOut[2] = dLdC[2] * w * sb * (1.0 - sb);
        dOut[3] = dSigma * softplus_deriv(raw[3]);
        g = nn.backward(net, caches[i], dOut);
        acc = nn.add_grads(accW, accB, g[0], g[1]);
        accW = acc[0];
        accB = acc[1];
        # maintain suffix: add this sample's own weighted contribution
        suffix = suffix + dot * w;
    }
    return [accW, accB];
}

# ---------------------------------------------------------------------------
# Camera rays: pinhole looking from eye at target, vertical fov in degrees.
# Returns [origins, dirs] as arrays of 3-vectors, row-major over the image.
# ---------------------------------------------------------------------------

fn camera_rays(eye, target, up, fovDeg, w, h) {
    int d = 0; int dirs = 0; int fwd = 0; int halfH = 0; int halfW = 0; int origins = 0; int right = 0; int trueUp = 0; int u = 0; int v = 0; int x = 0; int y = 0;
    fwd = vector.vnormalize(vector.vsub(target, eye));
    right = vector.vnormalize(vector.vcross(fwd, up));
    trueUp = vector.vcross(right, fwd);
    halfH = tan(mathx.rad(fovDeg) * 0.5);
    halfW = halfH * w / h;
    origins = [];
    dirs = [];
    for (y = 0; y < h; ++y) {
        v = (1.0 - 2.0 * (y + 0.5) / h) * halfH;
        for (x = 0; x < w; ++x) {
            u = (2.0 * (x + 0.5) / w - 1.0) * halfW;
            d = [fwd[0] + right[0] * u + trueUp[0] * v,
                 fwd[1] + right[1] * u + trueUp[1] * v,
                 fwd[2] + right[2] * u + trueUp[2] * v];
            dirs[len(dirs)] = vector.vnormalize(d);
            origins[len(origins)] = eye;
        }
    }
    return [origins, dirs];
}

# ---------------------------------------------------------------------------
# Ground-truth toy scene: a soft sphere whose color varies with position.
# Used to synthesize training images.
# ---------------------------------------------------------------------------

fn scene_query(p) {
    int col = 0; int r = 0; int sigma = 0;
    r = vector.vnorm(p);
    sigma = 0.0;
    if (r < 0.6) { sigma = 14.0 * (0.6 - r) / 0.6; }
    col = [0.5 + 0.8 * p[0], 0.5 + 0.8 * p[1], 0.9 - 0.6 * p[2]];
    return [mathx.clampf(col[0], 0.0, 1.0),
            mathx.clampf(col[1], 0.0, 1.0),
            mathx.clampf(col[2], 0.0, 1.0), sigma];
}

fn scene_render_ray(ro, rd, near, far, nSamples, bg) {
    int a = 0; int cb = 0; int cg = 0; int cr = 0; int dt = 0; int i = 0; int p = 0; int q = 0; int t = 0; int trans = 0; int wgt = 0;
    dt = (far - near) / nSamples;
    trans = 1.0;
    cr = 0.0; cg = 0.0; cb = 0.0;
    for (i = 0; i < nSamples; ++i) {
        t = near + dt * (i + 0.5);
        p = [ro[0] + rd[0] * t, ro[1] + rd[1] * t, ro[2] + rd[2] * t];
        q = scene_query(p);
        a = 1.0 - exp(0.0 - q[3] * dt);
        wgt = trans * a;
        cr = cr + wgt * q[0];
        cg = cg + wgt * q[1];
        cb = cb + wgt * q[2];
        trans = trans * (1.0 - a);
    }
    return [cr + trans * bg[0], cg + trans * bg[1], cb + trans * bg[2]];
}

# Render the ground-truth scene into an image (training data).
fn scene_render_image(eye, target, fovDeg, w, h, nSamples, bg) {
    int c = 0; int d = 0; int i = 0; int rays = 0; int rds = 0; int ros = 0;
    rays = camera_rays(eye, target, [0.0, 1.0, 0.0], fovDeg, w, h);
    ros = rays[0];
    rds = rays[1];
    d = [];
    for (i = 0; i < w * h; ++i) {
        c = scene_render_ray(ros[i], rds[i], 1.0, 4.2, nSamples, bg);
        d[i * 3] = c[0];
        d[i * 3 + 1] = c[1];
        d[i * 3 + 2] = c[2];
    }
    return [w, h, 3, d];
}

# Render the LEARNED field into an image.
fn field_render_image(field, eye, target, fovDeg, w, h, nSamples, bg) {
    int c = 0; int d = 0; int i = 0; int r = 0; int rays = 0; int rds = 0; int ros = 0;
    rays = camera_rays(eye, target, [0.0, 1.0, 0.0], fovDeg, w, h);
    ros = rays[0];
    rds = rays[1];
    d = [];
    for (i = 0; i < w * h; ++i) {
        r = render_ray(field, ros[i], rds[i], 1.0, 4.2, nSamples, bg);
        c = r[0];
        d[i * 3] = image.clamp01(c[0]);
        d[i * 3 + 1] = image.clamp01(c[1]);
        d[i * 3 + 2] = image.clamp01(c[2]);
    }
    return [w, h, 3, d];
}

# ---------------------------------------------------------------------------
# Training: one epoch over a batch of rays (SGD-style with any nn optimizer).
# rays = [origins, dirs], targets = flat rgb array (3 per ray).
# Returns [field, opt, meanLoss].
# ---------------------------------------------------------------------------

fn train_rays(field, opt, origins, dirs, targets, near, far, nSamples, bg) {
    int L = 0; int acc = 0; int accB = 0; int accW = 0; int aux = 0; int c = 0; int dLdC = 0; int eb = 0; int eg = 0; int er = 0; int g = 0; int i = 0; int n = 0; int net = 0; int r = 0; int sc = 0; int step = 0; int tb = 0; int tg = 0; int totalLoss = 0; int tr = 0; int z = 0;
    net = field[0];
    L = field[1];
    n = len(origins);
    z = nn.zero_grads(net);
    accW = z[0];
    accB = z[1];
    totalLoss = 0.0;
    for (i = 0; i < n; ++i) {
        r = render_ray(field, origins[i], dirs[i], near, far, nSamples, bg);
        c = r[0];
        aux = r[1];
        tr = targets[i * 3];
        tg = targets[i * 3 + 1];
        tb = targets[i * 3 + 2];
        er = c[0] - tr;
        eg = c[1] - tg;
        eb = c[2] - tb;
        totalLoss = totalLoss + (er * er + eg * eg + eb * eb) / 3.0;
        dLdC = [2.0 * er / 3.0, 2.0 * eg / 3.0, 2.0 * eb / 3.0];
        g = ray_grads(field, aux, dLdC);
        acc = nn.add_grads(accW, accB, g[0], g[1]);
        accW = acc[0];
        accB = acc[1];
    }
    sc = nn.scale_grads(accW, accB, 1.0 / n);
    step = nn.apply_grads(net, opt, sc[0], sc[1]);
    return [[step[0], L], step[1], totalLoss / n];
}

# PSNR between two same-size images (quality metric, higher is better).
fn psnr(a, b) {
    int da = 0; int db = 0; int dv = 0; int i = 0; int mse = 0;
    da = a[3];
    db = b[3];
    mse = 0.0;
    for (i = 0; i < len(da); ++i) {
        dv = da[i] - db[i];
        mse = mse + dv * dv;
    }
    mse = mse / len(da);
    if (mse < 1.0e-12) { return 99.0; }
    return 10.0 * log(1.0 / mse);
}
