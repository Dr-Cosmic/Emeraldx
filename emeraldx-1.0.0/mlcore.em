# =============================================================================
# mlcore.em - metrics, losses, distances and activations shared by the
# machine-learning modules
# =============================================================================

import mathx;
import arrayx;
import vector;
# emx-scope-safe

# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

fn euclidean(a, b) { return vector.vdist(a, b); }
fn sq_euclidean(a, b) { return vector.vdist_sq(a, b); }

fn manhattan(a, b) {
    int i = 0; int s = 0;
    s = 0.0;
    for (i = 0; i < len(a); ++i) { s = s + abs(a[i] - b[i]); }
    return s;
}

fn chebyshev(a, b) {
    int d = 0; int i = 0; int s = 0;
    s = 0.0;
    for (i = 0; i < len(a); ++i) {
        d = abs(a[i] - b[i]);
        if (d > s) { s = d; }
    }
    return s;
}

fn minkowski(a, b, p) {
    int i = 0; int s = 0;
    s = 0.0;
    for (i = 0; i < len(a); ++i) { s = s + abs(a[i] - b[i]) ** p; }
    return s ** (1.0 / p);
}

fn cosine_similarity(a, b) {
    int na = 0; int nb = 0;
    na = vector.vnorm(a);
    nb = vector.vnorm(b);
    if (na < 1.0e-12 || nb < 1.0e-12) { return 0.0; }
    return vector.vdot(a, b) / (na * nb);
}

fn cosine_distance(a, b) { return 1.0 - cosine_similarity(a, b); }

# ---------------------------------------------------------------------------
# Activations (scalar; nn.em applies them over arrays)
# ---------------------------------------------------------------------------

fn sigmoid(x) { return mathx.logistic(x); }
fn sigmoid_deriv(y) { return y * (1.0 - y); }        # takes the OUTPUT y

fn relu(x) {
    if (x > 0.0) { return x; }
    return 0.0;
}
fn relu_deriv(x) {
    if (x > 0.0) { return 1.0; }
    return 0.0;
}

fn leaky_relu(x) {
    if (x > 0.0) { return x; }
    return 0.01 * x;
}
fn leaky_relu_deriv(x) {
    if (x > 0.0) { return 1.0; }
    return 0.01;
}

fn tanh_act(x) { return tanh(x); }
fn tanh_deriv(y) { return 1.0 - y * y; }             # takes the OUTPUT y

# Numerically-stable softmax over an array.
fn softmax(xs) {
    int i = 0; int m = 0; int out = 0; int s = 0; int v = 0;
    m = arrayx.maxv(xs);
    out = [];
    s = 0.0;
    for (i = 0; i < len(xs); ++i) {
        v = exp(xs[i] - m);
        out[i] = v;
        s = s + v;
    }
    for (i = 0; i < len(xs); ++i) { out[i] = out[i] / s; }
    return out;
}

# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

fn mse(pred, target) {
    int d = 0; int i = 0; int s = 0;
    s = 0.0;
    for (i = 0; i < len(pred); ++i) {
        d = pred[i] - target[i];
        s = s + d * d;
    }
    return s / len(pred);
}

fn mae(pred, target) {
    int i = 0; int s = 0;
    s = 0.0;
    for (i = 0; i < len(pred); ++i) { s = s + abs(pred[i] - target[i]); }
    return s / len(pred);
}

# Binary cross-entropy; pred are probabilities.
fn bce(pred, target) {
    int i = 0; int p = 0; int s = 0;
    s = 0.0;
    for (i = 0; i < len(pred); ++i) {
        p = mathx.clampf(pred[i], 1.0e-9, 1.0 - 1.0e-9);
        s = s - (target[i] * ln(p) + (1.0 - target[i]) * ln(1.0 - p));
    }
    return s / len(pred);
}

# Categorical cross-entropy for one sample; pred is a softmax row,
# target the class index.
fn cce_row(pred, classIdx) {
    int p = 0;
    p = mathx.clampf(pred[classIdx], 1.0e-9, 1.0);
    return 0.0 - ln(p);
}

# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

fn r2_score(pred, target) {
    int i = 0; int m = 0; int n = 0; int ssRes = 0; int ssTot = 0;
    n = len(target);
    m = 0.0;
    for y in target { m = m + y; }
    m = m / n;
    ssRes = 0.0;
    ssTot = 0.0;
    for (i = 0; i < n; ++i) {
        ssRes = ssRes + (target[i] - pred[i]) * (target[i] - pred[i]);
        ssTot = ssTot + (target[i] - m) * (target[i] - m);
    }
    if (ssTot < 1.0e-12) { return 0.0; }
    return 1.0 - ssRes / ssTot;
}

fn rmse(pred, target) { return sqrt(mse(pred, target)); }

# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

fn accuracy(pred, target) {
    int c = 0; int i = 0;
    if (len(pred) == 0) { return 0.0; }
    c = 0;
    for (i = 0; i < len(pred); ++i) {
        if (pred[i] == target[i]) { c = c + 1; }
    }
    return c * 1.0 / len(pred);
}

# k x k confusion matrix; rows = actual class, cols = predicted class.
fn confusion_matrix(pred, target, k) {
    int i = 0; int m = 0; int row = 0;
    m = [];
    for (i = 0; i < k; ++i) { m[i] = arrayx.zerosf(k); }
    for (i = 0; i < len(pred); ++i) {
        row = m[target[i]];
        row[pred[i]] = row[pred[i]] + 1;
        m[target[i]] = row;
    }
    return m;
}

# Per-class precision/recall/F1 from a confusion matrix.
# Returns [precisions, recalls, f1s].
fn precision_recall_f1(cm) {
    int c = 0; int f = 0; int f1s = 0; int fnc = 0; int fp = 0; int i = 0; int k = 0; int p = 0; int precs = 0; int r = 0; int recs = 0; int tp = 0;
    k = len(cm);
    precs = [];
    recs = [];
    f1s = [];
    for (c = 0; c < k; ++c) {
        tp = cm[c][c];
        fp = 0.0;
        fnc = 0.0;
        for (i = 0; i < k; ++i) {
            if (i != c) {
                fp = fp + cm[i][c];
                fnc = fnc + cm[c][i];
            }
        }
        p = 0.0;
        if (tp + fp > 0) { p = tp / (tp + fp); }
        r = 0.0;
        if (tp + fnc > 0) { r = tp / (tp + fnc); }
        f = 0.0;
        if (p + r > 1.0e-12) { f = 2.0 * p * r / (p + r); }
        precs[c] = p;
        recs[c] = r;
        f1s[c] = f;
    }
    return [precs, recs, f1s];
}

# Argmax over class-score rows -> predicted labels.
fn predict_labels(scoreRows) {
    int i = 0; int out = 0;
    out = [];
    for (i = 0; i < len(scoreRows); ++i) { out[i] = arrayx.argmax(scoreRows[i]); }
    return out;
}
