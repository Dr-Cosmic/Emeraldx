# =============================================================================
# nn.em - feed-forward neural networks with manual backpropagation
#
# A network is net = [sizes, acts, weights, biases]:
#   sizes   [in, h1, ..., out]
#   acts    per-layer activation names: "relu" | "tanh" | "sigmoid" | "linear"
#   weights array of (out x in) nested matrices
#   biases  array of arrays
#
# Training state for the optimizer is threaded separately (opt value).
# A full forward pass keeps every layer's pre-activation and activation so
# gradients can also be requested with respect to the INPUT - that is what
# nerf.em uses to differentiate through volume rendering.
#
#   net = nn.build([2, 8, 1], ["tanh", "sigmoid"], seed)   -> [net, seed]
#   fwd = nn.forward(net, x)             -> cache
#   y   = nn.output(fwd)
#   g   = nn.backward(net, fwd, dLdOut)  -> [gradW, gradB, dLdIn]
#   net = nn.apply_grads(net, opt, gradW, gradB)[0] ...
# =============================================================================

import mathx;
import arrayx;
import matrix;
import prng;
import mlcore;
# emx-scope-safe

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

# Xavier/He initialized network. Returns [net, newSeed].
fn build(sizes, acts, seed) {
    int bs = 0; int fanIn = 0; int fanOut = 0; int l = 0; int nl = 0; int r = 0; int sc = 0; int ws = 0;
    nl = len(sizes) - 1;
    ws = [];
    bs = [];
    for (l = 0; l < nl; ++l) {
        fanIn = sizes[l];
        fanOut = sizes[l + 1];
        sc = sqrt(2.0 / fanIn);                     # He (good for relu)
        if (acts[l] == "tanh" || acts[l] == "sigmoid") {
            sc = sqrt(1.0 / fanIn);                 # Xavier-ish
        }
        r = matrix.randn(fanOut, fanIn, seed, sc);
        ws[l] = r[0];
        seed = r[1];
        bs[l] = arrayx.zerosf(fanOut);
    }
    return [[sizes, acts, ws, bs], seed];
}

fn act_apply(name, z) {
    int i = 0; int out = 0;
    out = [];
    if (name == "relu") {
        for (i = 0; i < len(z); ++i) { out[i] = mlcore.relu(z[i]); }
    } elif (name == "tanh") {
        for (i = 0; i < len(z); ++i) { out[i] = tanh(z[i]); }
    } elif (name == "sigmoid") {
        for (i = 0; i < len(z); ++i) { out[i] = mathx.logistic(z[i]); }
    } else {
        out = z;
    }
    return out;
}

# Derivative of the activation given pre-activation z and activation a.
fn act_deriv(name, z, a) {
    int i = 0; int out = 0;
    out = [];
    if (name == "relu") {
        for (i = 0; i < len(z); ++i) { out[i] = mlcore.relu_deriv(z[i]); }
    } elif (name == "tanh") {
        for (i = 0; i < len(a); ++i) { out[i] = 1.0 - a[i] * a[i]; }
    } elif (name == "sigmoid") {
        for (i = 0; i < len(a); ++i) { out[i] = a[i] * (1.0 - a[i]); }
    } else {
        for (i = 0; i < len(z); ++i) { out[i] = 1.0; }
    }
    return out;
}

# ---------------------------------------------------------------------------
# Forward / backward
# ---------------------------------------------------------------------------

# Full forward pass. Returns cache = [activations, preacts]:
# activations[0] is the input, activations[nl] the network output.
fn forward(net, x) {
    int a = 0; int activations = 0; int acts = 0; int bl = 0; int bs = 0; int i = 0; int l = 0; int nl = 0; int preacts = 0; int ws = 0; int z = 0;
    acts = net[1];
    ws = net[2];
    bs = net[3];
    nl = len(ws);
    activations = [];
    preacts = [];
    activations[0] = x;
    a = x;
    for (l = 0; l < nl; ++l) {
        z = matrix.matvec(ws[l], a);
        bl = bs[l];
        for (i = 0; i < len(z); ++i) { z[i] = z[i] + bl[i]; }
        preacts[l] = z;
        a = act_apply(acts[l], z);
        activations[l + 1] = a;
    }
    return [activations, preacts];
}

fn output(cache) {
    int activations = 0;
    activations = cache[0];
    return activations[len(activations) - 1];
}

# Fast inference without cache.
fn predict(net, x) {
    return output(forward(net, x));
}

# Backprop from an arbitrary dL/dOut vector.
# Returns [gradW, gradB, dLdInput].
fn backward(net, cache, dOut) {
    int aPrev = 0; int activations = 0; int acts = 0; int delta = 0; int di = 0; int dz = 0; int gW = 0; int gradB = 0; int gradW = 0; int i = 0; int j = 0; int l = 0; int nd = 0; int nl = 0; int nprev = 0; int preacts = 0; int rowg = 0; int rw = 0; int wl = 0; int ws = 0;
    acts = net[1];
    ws = net[2];
    activations = cache[0];
    preacts = cache[1];
    nl = len(ws);
    gradW = [];
    gradB = [];
    delta = dOut;
    for (l = nl - 1; l >= 0; --l) {
        dz = act_deriv(acts[l], preacts[l], activations[l + 1]);
        for (i = 0; i < len(delta); ++i) { delta[i] = delta[i] * dz[i]; }
        gradB[l] = delta;
        aPrev = activations[l];
        gW = [];
        for (i = 0; i < len(delta); ++i) {
            rowg = [];
            di = delta[i];
            for (j = 0; j < len(aPrev); ++j) { rowg[j] = di * aPrev[j]; }
            gW[i] = rowg;
        }
        gradW[l] = gW;
        # propagate: delta_prev = W^T delta
        wl = ws[l];
        nprev = len(aPrev);
        nd = [];
        for (j = 0; j < nprev; ++j) { nd[j] = 0.0; }
        for (i = 0; i < len(delta); ++i) {
            di = delta[i];
            if (di != 0.0) {
                rw = wl[i];
                for (j = 0; j < nprev; ++j) { nd[j] = nd[j] + rw[j] * di; }
            }
        }
        delta = nd;
    }
    return [gradW, gradB, delta];
}

# Zero-valued gradient accumulators matching a network.
fn zero_grads(net) {
    int gb = 0; int gw = 0; int l = 0; int nl = 0; int sizes = 0;
    sizes = net[0];
    nl = len(sizes) - 1;
    gw = [];
    gb = [];
    for (l = 0; l < nl; ++l) {
        gw[l] = matrix.zeros(sizes[l + 1], sizes[l]);
        gb[l] = arrayx.zerosf(sizes[l + 1]);
    }
    return [gw, gb];
}

fn add_grads(accW, accB, gw, gb) {
    int al = 0; int gl = 0; int i = 0; int l = 0;
    for (l = 0; l < len(gw); ++l) {
        accW[l] = matrix.add(accW[l], gw[l]);
        al = accB[l];
        gl = gb[l];
        for (i = 0; i < len(al); ++i) { al[i] = al[i] + gl[i]; }
        accB[l] = al;
    }
    return [accW, accB];
}

fn scale_grads(gw, gb, s) {
    int gl = 0; int i = 0; int l = 0;
    for (l = 0; l < len(gw); ++l) {
        gw[l] = matrix.scale(gw[l], s);
        gl = gb[l];
        for (i = 0; i < len(gl); ++i) { gl[i] = gl[i] * s; }
        gb[l] = gl;
    }
    return [gw, gb];
}

# ---------------------------------------------------------------------------
# Optimizers: opt = ["sgd", lr] | ["momentum", lr, mu, velW, velB]
#                 | ["adam", lr, b1, b2, t, mW, mB, vW, vB]
# ---------------------------------------------------------------------------

fn opt_sgd(lr) { return ["sgd", lr]; }

fn opt_momentum(net, lr, mu) {
    int z = 0;
    z = zero_grads(net);
    return ["momentum", lr, mu, z[0], z[1]];
}

fn opt_adam(net, lr) {
    int z1 = 0; int z2 = 0;
    z1 = zero_grads(net);
    z2 = zero_grads(net);
    return ["adam", lr, 0.9, 0.999, 0, z1[0], z1[1], z2[0], z2[1]];
}

# Apply gradients. Returns [net, opt].
fn apply_grads(net, opt, gw, gb) {
    int b1 = 0; int b2 = 0; int bc1 = 0; int bc2 = 0; int bl = 0; int bs = 0; int gl = 0; int i = 0; int j = 0; int kind = 0; int l = 0; int lr = 0; int mb = 0; int mbl = 0; int mhat = 0; int mr = 0; int mu = 0; int mw = 0; int mwl = 0; int nl = 0; int t = 0; int vb = 0; int vbl = 0; int vhat = 0; int vr = 0; int vw = 0; int vwl = 0; int wl = 0; int wr = 0; int ws = 0;
    ws = net[2];
    bs = net[3];
    kind = opt[0];
    nl = len(ws);
    if (kind == "sgd") {
        lr = opt[1];
        for (l = 0; l < nl; ++l) {
            ws[l] = matrix.sub(ws[l], matrix.scale(gw[l], lr));
            bl = bs[l];
            gl = gb[l];
            for (i = 0; i < len(bl); ++i) { bl[i] = bl[i] - lr * gl[i]; }
            bs[l] = bl;
        }
    } elif (kind == "momentum") {
        lr = opt[1];
        mu = opt[2];
        vw = opt[3];
        vb = opt[4];
        for (l = 0; l < nl; ++l) {
            vw[l] = matrix.sub(matrix.scale(vw[l], mu), matrix.scale(gw[l], lr));
            ws[l] = matrix.add(ws[l], vw[l]);
            vbl = vb[l];
            gl = gb[l];
            bl = bs[l];
            for (i = 0; i < len(bl); ++i) {
                vbl[i] = mu * vbl[i] - lr * gl[i];
                bl[i] = bl[i] + vbl[i];
            }
            vb[l] = vbl;
            bs[l] = bl;
        }
        opt = ["momentum", lr, mu, vw, vb];
    } else {
        # adam
        lr = opt[1];
        b1 = opt[2];
        b2 = opt[3];
        t = opt[4] + 1;
        mw = opt[5];
        mb = opt[6];
        vw = opt[7];
        vb = opt[8];
        bc1 = 1.0 - b1 ** t;
        bc2 = 1.0 - b2 ** t;
        for (l = 0; l < nl; ++l) {
            mw[l] = matrix.add(matrix.scale(mw[l], b1), matrix.scale(gw[l], 1.0 - b1));
            vw[l] = matrix.add(matrix.scale(vw[l], b2),
                               matrix.scale(matrix.hadamard(gw[l], gw[l]), 1.0 - b2));
            wl = ws[l];
            mwl = mw[l];
            vwl = vw[l];
            for (i = 0; i < len(wl); ++i) {
                wr = wl[i];
                mr = mwl[i];
                vr = vwl[i];
                for (j = 0; j < len(wr); ++j) {
                    mhat = mr[j] / bc1;
                    vhat = vr[j] / bc2;
                    wr[j] = wr[j] - lr * mhat / (sqrt(vhat) + 1.0e-8);
                }
                wl[i] = wr;
            }
            ws[l] = wl;
            mbl = mb[l];
            vbl = vb[l];
            gl = gb[l];
            bl = bs[l];
            for (i = 0; i < len(bl); ++i) {
                mbl[i] = b1 * mbl[i] + (1.0 - b1) * gl[i];
                vbl[i] = b2 * vbl[i] + (1.0 - b2) * gl[i] * gl[i];
                mhat = mbl[i] / bc1;
                vhat = vbl[i] / bc2;
                bl[i] = bl[i] - lr * mhat / (sqrt(vhat) + 1.0e-8);
            }
            mb[l] = mbl;
            vb[l] = vbl;
            bs[l] = bl;
        }
        opt = ["adam", lr, b1, b2, t, mw, mb, vw, vb];
    }
    return [[net[0], net[1], ws, bs], opt];
}

# ---------------------------------------------------------------------------
# High-level training
# ---------------------------------------------------------------------------

# dL/dOut for common losses (single sample).
fn loss_grad(lossName, pred, target) {
    int g = 0; int i = 0; int n = 0; int p = 0;
    g = [];
    n = len(pred);
    if (lossName == "mse") {
        for (i = 0; i < n; ++i) { g[i] = 2.0 * (pred[i] - target[i]) / n; }
    } elif (lossName == "bce") {
        for (i = 0; i < n; ++i) {
            p = mathx.clampf(pred[i], 1.0e-9, 1.0 - 1.0e-9);
            g[i] = (p - target[i]) / (p * (1.0 - p)) / n;
        }
    } else {
        for (i = 0; i < n; ++i) { g[i] = pred[i] - target[i]; }
    }
    return g;
}

fn loss_value(lossName, pred, target) {
    if (lossName == "bce") { return mlcore.bce(pred, target); }
    return mlcore.mse(pred, target);
}

# Full-batch training loop.
# xs: array of input arrays; ts: array of target arrays.
# Returns [net, opt, lossHistory].
fn train(net, opt, xs, ts, lossName, epochs, verboseEvery) {
    int acc = 0; int accB = 0; int accW = 0; int avgLoss = 0; int cache = 0; int ep = 0; int g = 0; int history = 0; int i = 0; int n = 0; int pred = 0; int sc = 0; int step = 0; int totalLoss = 0; int z = 0;
    n = len(xs);
    history = [];
    for (ep = 0; ep < epochs; ++ep) {
        z = zero_grads(net);
        accW = z[0];
        accB = z[1];
        totalLoss = 0.0;
        for (i = 0; i < n; ++i) {
            cache = forward(net, xs[i]);
            pred = output(cache);
            totalLoss = totalLoss + loss_value(lossName, pred, ts[i]);
            g = backward(net, cache, loss_grad(lossName, pred, ts[i]));
            acc = add_grads(accW, accB, g[0], g[1]);
            accW = acc[0];
            accB = acc[1];
        }
        sc = scale_grads(accW, accB, 1.0 / n);
        step = apply_grads(net, opt, sc[0], sc[1]);
        net = step[0];
        opt = step[1];
        avgLoss = totalLoss / n;
        history[len(history)] = avgLoss;
        if (verboseEvery != nil && verboseEvery > 0 && ep % verboseEvery == 0) {
            print("epoch ", ep, "  loss ", avgLoss);
        }
    }
    return [net, opt, history];
}

# Classifier helper: net outputs raw scores; returns predicted class index.
fn classify(net, x) {
    return arrayx.argmax(predict(net, x));
}
