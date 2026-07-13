# =============================================================================
# cnn.em - small convolutional building blocks
#
# Feature maps are [h, w, data] with data flat row-major (single channel);
# multi-channel stacks are arrays of such maps. Includes forward AND backward
# passes for conv2d (valid padding) and 2x2 maxpool, enough to train a tiny
# convnet with the optimizers from nn.em.
# =============================================================================

import mathx;
import arrayx;
import prng;
# emx-scope-safe

fn fmap(h, w, fill) {
    if (fill == nil) { fill = 0.0; }
    return [h, w, arrayx.full(h * w, fill)];
}

fn fmap_from(h, w, data) { return [h, w, data]; }

fn fget(m, y, x) {
    int d = 0;
    d = m[2];
    return d[y * m[1] + x];
}

# Random kernel (kh x kw), He-scaled. Returns [kernel, newSeed].
fn kernel_rand(kh, kw, seed) {
    int d = 0; int i = 0; int r = 0; int sc = 0;
    r = prng.normal_array(seed, kh * kw);
    d = r[0];
    sc = sqrt(2.0 / (kh * kw));
    for (i = 0; i < len(d); ++i) { d[i] = d[i] * sc; }
    return [[kh, kw, d], r[1]];
}

# Valid convolution (technically cross-correlation, as in every DL library).
fn conv2d(input, kernel, bias) {
    int ibase = 0; int idata = 0; int ih = 0; int iw = 0; int kbase = 0; int kdata = 0; int kh = 0; int kw = 0; int kx = 0; int ky = 0; int oh = 0; int out = 0; int ow = 0; int ox = 0; int oy = 0; int s = 0;
    if (bias == nil) { bias = 0.0; }
    ih = input[0];
    iw = input[1];
    idata = input[2];
    kh = kernel[0];
    kw = kernel[1];
    kdata = kernel[2];
    oh = ih - kh + 1;
    ow = iw - kw + 1;
    out = [];
    for (oy = 0; oy < oh; ++oy) {
        for (ox = 0; ox < ow; ++ox) {
            s = bias;
            for (ky = 0; ky < kh; ++ky) {
                ibase = (oy + ky) * iw + ox;
                kbase = ky * kw;
                for (kx = 0; kx < kw; ++kx) {
                    s = s + idata[ibase + kx] * kdata[kbase + kx];
                }
            }
            out[oy * ow + ox] = s;
        }
    }
    return [oh, ow, out];
}

# Gradients of a conv2d layer.
# dOut: [oh, ow, ...] gradient at the output.
# Returns [dKernel, dBias, dInput].
fn conv2d_backward(input, kernel, dOut) {
    int dbias = 0; int dinp = 0; int dk = 0; int g = 0; int gdata = 0; int ibase = 0; int idata = 0; int ih = 0; int iw = 0; int kbase = 0; int kdata = 0; int kh = 0; int kw = 0; int kx = 0; int ky = 0; int oh = 0; int ow = 0; int ox = 0; int oy = 0;
    ih = input[0];
    iw = input[1];
    idata = input[2];
    kh = kernel[0];
    kw = kernel[1];
    kdata = kernel[2];
    oh = dOut[0];
    ow = dOut[1];
    gdata = dOut[2];
    dk = arrayx.zerosf(kh * kw);
    dbias = 0.0;
    dinp = arrayx.zerosf(ih * iw);
    for (oy = 0; oy < oh; ++oy) {
        for (ox = 0; ox < ow; ++ox) {
            g = gdata[oy * ow + ox];
            if (g == 0.0) { continue; }
            dbias = dbias + g;
            for (ky = 0; ky < kh; ++ky) {
                ibase = (oy + ky) * iw + ox;
                kbase = ky * kw;
                for (kx = 0; kx < kw; ++kx) {
                    dk[kbase + kx] = dk[kbase + kx] + g * idata[ibase + kx];
                    dinp[ibase + kx] = dinp[ibase + kx] + g * kdata[kbase + kx];
                }
            }
        }
    }
    return [[kh, kw, dk], dbias, [ih, iw, dinp]];
}

# ReLU on a feature map. Returns [out, mask] (mask reused by the backward).
fn relu_fmap(m) {
    int d = 0; int i = 0; int mask = 0; int out = 0;
    d = m[2];
    out = [];
    mask = [];
    for (i = 0; i < len(d); ++i) {
        if (d[i] > 0.0) { out[i] = d[i]; mask[i] = 1.0; }
        else { out[i] = 0.0; mask[i] = 0.0; }
    }
    return [[m[0], m[1], out], [m[0], m[1], mask]];
}

fn relu_fmap_backward(dOut, mask) {
    int d = 0; int i = 0; int mk = 0; int out = 0;
    d = dOut[2];
    mk = mask[2];
    out = [];
    for (i = 0; i < len(d); ++i) { out[i] = d[i] * mk[i]; }
    return [dOut[0], dOut[1], out];
}

# 2x2 max pooling with stride 2 (input dims should be even).
# Returns [pooled, argmaxIdx] where argmaxIdx stores the flat input index of
# each pooled maximum (for the backward pass).
fn maxpool2(m) {
    int argm = 0; int best = 0; int bi = 0; int d = 0; int i0 = 0; int i1 = 0; int i2 = 0; int i3 = 0; int ih = 0; int iw = 0; int oh = 0; int out = 0; int ow = 0; int ox = 0; int oy = 0;
    ih = m[0];
    iw = m[1];
    d = m[2];
    oh = ih // 2;
    ow = iw // 2;
    out = [];
    argm = [];
    for (oy = 0; oy < oh; ++oy) {
        for (ox = 0; ox < ow; ++ox) {
            i0 = (2 * oy) * iw + 2 * ox;
            i1 = i0 + 1;
            i2 = i0 + iw;
            i3 = i2 + 1;
            best = d[i0];
            bi = i0;
            if (d[i1] > best) { best = d[i1]; bi = i1; }
            if (d[i2] > best) { best = d[i2]; bi = i2; }
            if (d[i3] > best) { best = d[i3]; bi = i3; }
            out[oy * ow + ox] = best;
            argm[oy * ow + ox] = bi;
        }
    }
    return [[oh, ow, out], argm];
}

fn maxpool2_backward(dOut, argm, ih, iw) {
    int dinp = 0; int g = 0; int i = 0;
    g = dOut[2];
    dinp = arrayx.zerosf(ih * iw);
    for (i = 0; i < len(g); ++i) {
        dinp[argm[i]] = dinp[argm[i]] + g[i];
    }
    return [ih, iw, dinp];
}

fn flatten_fmap(m) { return m[2]; }

fn unflatten_fmap(h, w, flat) { return [h, w, flat]; }

# Average pool the whole map to one number (global average pooling).
fn global_avg(m) {
    int d = 0; int s = 0;
    d = m[2];
    s = 0.0;
    for x in d { s = s + x; }
    return s / len(d);
}
