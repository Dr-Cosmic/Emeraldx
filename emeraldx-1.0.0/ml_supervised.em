# =============================================================================
# ml_supervised.em - classic supervised learning
#
# Models are plain arrays; every trainer returns a model value and every
# predictor takes (model, x). Nothing is mutated.
#
#   linreg_fit / linreg_predict          least squares (QR)
#   ridge_fit                            L2-regularized least squares
#   linreg_gd_fit                        gradient descent variant
#   logistic_fit / logistic_predict      binary classifier
#   knn_predict                          k-nearest neighbours (lazy)
#   gnb_fit / gnb_predict                gaussian naive bayes
#   perceptron_fit / perceptron_predict
#   tree_fit / tree_predict              CART decision tree (gini / mse)
# =============================================================================

import mathx;
import arrayx;
import vector;
import matrix;
import mlcore;
# emx-scope-safe

# ---------------------------------------------------------------------------
# Linear regression (closed form): model = [w, b]
# ---------------------------------------------------------------------------

fn linreg_fit(xs, ys) {
    int a = 0; int d = 0; int i = 0; int j = 0; int n = 0; int w = 0; int wb = 0;
    n = len(xs);
    d = len(xs[0]);
    a = [];
    for (i = 0; i < n; ++i) { a[i] = xs[i] + [1.0]; }   # bias column
    wb = matrix.lstsq(a, ys);
    w = [];
    for (j = 0; j < d; ++j) { w[j] = wb[j]; }
    return [w, wb[d]];
}

# Ridge regression: (X^T X + lambda I) w = X^T y  (bias not regularized).
fn ridge_fit(xs, ys, lambda) {
    int a = 0; int at = 0; int ata = 0; int aty = 0; int d = 0; int i = 0; int j = 0; int n = 0; int w = 0; int wb = 0;
    n = len(xs);
    d = len(xs[0]);
    a = [];
    for (i = 0; i < n; ++i) { a[i] = xs[i] + [1.0]; }
    at = matrix.transpose(a);
    ata = matrix.matmul(at, a);
    for (j = 0; j < d; ++j) { ata[j][j] = ata[j][j] + lambda; }
    aty = matrix.matvec(at, ys);
    wb = matrix.solve(ata, aty);
    if (wb == nil) { return linreg_fit(xs, ys); }
    w = [];
    for (j = 0; j < d; ++j) { w[j] = wb[j]; }
    return [w, wb[d]];
}

fn linreg_predict_one(model, x) {
    return vector.vdot(model[0], x) + model[1];
}

fn linreg_predict(model, xs) {
    int i = 0; int out = 0;
    out = [];
    for (i = 0; i < len(xs); ++i) { out[i] = linreg_predict_one(model, xs[i]); }
    return out;
}

# Gradient-descent linear regression (for large d or streaming style).
fn linreg_gd_fit(xs, ys, lr, epochs) {
    int b = 0; int d = 0; int ep = 0; int err = 0; int gb = 0; int gw = 0; int i = 0; int j = 0; int n = 0; int row = 0; int w = 0;
    if (lr == nil) { lr = 0.05; }
    if (epochs == nil) { epochs = 200; }
    n = len(xs);
    d = len(xs[0]);
    w = arrayx.zerosf(d);
    b = 0.0;
    for (ep = 0; ep < epochs; ++ep) {
        gw = arrayx.zerosf(d);
        gb = 0.0;
        for (i = 0; i < n; ++i) {
            row = xs[i];
            err = vector.vdot(w, row) + b - ys[i];
            for (j = 0; j < d; ++j) { gw[j] = gw[j] + err * row[j]; }
            gb = gb + err;
        }
        for (j = 0; j < d; ++j) { w[j] = w[j] - lr * 2.0 * gw[j] / n; }
        b = b - lr * 2.0 * gb / n;
    }
    return [w, b];
}

# ---------------------------------------------------------------------------
# Logistic regression (binary, labels 0/1): model = [w, b]
# ---------------------------------------------------------------------------

fn logistic_fit(xs, ys, lr, epochs, l2) {
    int b = 0; int d = 0; int ep = 0; int err = 0; int gb = 0; int gw = 0; int i = 0; int j = 0; int n = 0; int p = 0; int row = 0; int w = 0;
    if (lr == nil) { lr = 0.2; }
    if (epochs == nil) { epochs = 300; }
    if (l2 == nil) { l2 = 0.0; }
    n = len(xs);
    d = len(xs[0]);
    w = arrayx.zerosf(d);
    b = 0.0;
    for (ep = 0; ep < epochs; ++ep) {
        gw = arrayx.zerosf(d);
        gb = 0.0;
        for (i = 0; i < n; ++i) {
            row = xs[i];
            p = mathx.logistic(vector.vdot(w, row) + b);
            err = p - ys[i];
            for (j = 0; j < d; ++j) { gw[j] = gw[j] + err * row[j]; }
            gb = gb + err;
        }
        for (j = 0; j < d; ++j) {
            w[j] = w[j] - lr * (gw[j] / n + l2 * w[j]);
        }
        b = b - lr * gb / n;
    }
    return [w, b];
}

fn logistic_prob_one(model, x) {
    return mathx.logistic(vector.vdot(model[0], x) + model[1]);
}

fn logistic_predict(model, xs, threshold) {
    int i = 0; int out = 0; int p = 0;
    if (threshold == nil) { threshold = 0.5; }
    out = [];
    for (i = 0; i < len(xs); ++i) {
        p = logistic_prob_one(model, xs[i]);
        if (p >= threshold) { out[i] = 1; } else { out[i] = 0; }
    }
    return out;
}

# ---------------------------------------------------------------------------
# k-nearest neighbours (classification): lazy model = [X, y, k]
# ---------------------------------------------------------------------------

fn knn_model(xs, ys, k) { return [xs, ys, k]; }

fn knn_predict_one(model, x) {
    int c = 0; int dists = 0; int found = 0; int i = 0; int k = 0; int labels = 0; int lbl = 0; int n = 0; int order = 0; int t = 0; int votes = 0; int xs = 0; int ys = 0;
    xs = model[0];
    ys = model[1];
    k = model[2];
    n = len(xs);
    dists = [];
    for (i = 0; i < n; ++i) { dists[i] = mlcore.sq_euclidean(xs[i], x); }
    order = arrayx.argsort(dists);
    votes = [];
    labels = [];
    for (t = 0; t < k && t < n; ++t) {
        lbl = ys[order[t]];
        found = -1;
        for (c = 0; c < len(labels); ++c) {
            if (labels[c] == lbl) { found = c; break; }
        }
        if (found < 0) {
            labels[len(labels)] = lbl;
            votes[len(votes)] = 1;
        } else {
            votes[found] = votes[found] + 1;
        }
    }
    return labels[arrayx.argmax(votes)];
}

fn knn_predict(model, xs) {
    int i = 0; int out = 0;
    out = [];
    for (i = 0; i < len(xs); ++i) { out[i] = knn_predict_one(model, xs[i]); }
    return out;
}

# ---------------------------------------------------------------------------
# Gaussian naive bayes: model = [priors, means, vars, k]
# Labels must be integers 0..k-1.
# ---------------------------------------------------------------------------

fn gnb_fit(xs, ys, k) {
    int c = 0; int cc = 0; int counts = 0; int d = 0; int dv = 0; int i = 0; int j = 0; int means = 0; int mrow = 0; int n = 0; int priors = 0; int row = 0; int vars2 = 0; int vrow = 0;
    n = len(xs);
    d = len(xs[0]);
    counts = arrayx.zerosf(k);
    means = [];
    vars2 = [];
    for (c = 0; c < k; ++c) {
        means[c] = arrayx.zerosf(d);
        vars2[c] = arrayx.zerosf(d);
    }
    for (i = 0; i < n; ++i) {
        c = ys[i];
        counts[c] = counts[c] + 1;
        mrow = means[c];
        row = xs[i];
        for (j = 0; j < d; ++j) { mrow[j] = mrow[j] + row[j]; }
        means[c] = mrow;
    }
    for (c = 0; c < k; ++c) {
        mrow = means[c];
        cc = mathx.maxv(counts[c], 1.0);
        for (j = 0; j < d; ++j) { mrow[j] = mrow[j] / cc; }
        means[c] = mrow;
    }
    for (i = 0; i < n; ++i) {
        c = ys[i];
        vrow = vars2[c];
        mrow = means[c];
        row = xs[i];
        for (j = 0; j < d; ++j) {
            dv = row[j] - mrow[j];
            vrow[j] = vrow[j] + dv * dv;
        }
        vars2[c] = vrow;
    }
    priors = [];
    for (c = 0; c < k; ++c) {
        vrow = vars2[c];
        cc = mathx.maxv(counts[c], 1.0);
        for (j = 0; j < d; ++j) {
            vrow[j] = vrow[j] / cc + 1.0e-9;
        }
        vars2[c] = vrow;
        priors[c] = counts[c] / n;
    }
    return [priors, means, vars2, k];
}

fn gnb_log_prob(model, x, c) {
    int dv = 0; int j = 0; int lp = 0; int means = 0; int mrow = 0; int priors = 0; int vars2 = 0; int vrow = 0;
    priors = model[0];
    means = model[1];
    vars2 = model[2];
    lp = ln(mathx.maxv(priors[c], 1.0e-12));
    mrow = means[c];
    vrow = vars2[c];
    for (j = 0; j < len(x); ++j) {
        dv = x[j] - mrow[j];
        lp = lp - 0.5 * ln(mathx.TAU * vrow[j]) - dv * dv / (2.0 * vrow[j]);
    }
    return lp;
}

fn gnb_predict_one(model, x) {
    int best = 0; int bestLp = 0; int c = 0; int k = 0; int lp = 0;
    k = model[3];
    best = 0;
    bestLp = gnb_log_prob(model, x, 0);
    for (c = 1; c < k; ++c) {
        lp = gnb_log_prob(model, x, c);
        if (lp > bestLp) { bestLp = lp; best = c; }
    }
    return best;
}

fn gnb_predict(model, xs) {
    int i = 0; int out = 0;
    out = [];
    for (i = 0; i < len(xs); ++i) { out[i] = gnb_predict_one(model, xs[i]); }
    return out;
}

# ---------------------------------------------------------------------------
# Perceptron (labels 0/1): model = [w, b]
# ---------------------------------------------------------------------------

fn perceptron_fit(xs, ys, lr, epochs) {
    int b = 0; int d = 0; int ep = 0; int err = 0; int errs = 0; int i = 0; int j = 0; int pred = 0; int row = 0; int w = 0;
    if (lr == nil) { lr = 0.1; }
    if (epochs == nil) { epochs = 50; }
    d = len(xs[0]);
    w = arrayx.zerosf(d);
    b = 0.0;
    for (ep = 0; ep < epochs; ++ep) {
        errs = 0;
        for (i = 0; i < len(xs); ++i) {
            row = xs[i];
            pred = 0;
            if (vector.vdot(w, row) + b > 0.0) { pred = 1; }
            err = ys[i] - pred;
            if (err != 0) {
                errs = errs + 1;
                for (j = 0; j < d; ++j) { w[j] = w[j] + lr * err * row[j]; }
                b = b + lr * err;
            }
        }
        if (errs == 0) { break; }
    }
    return [w, b];
}

fn perceptron_predict(model, xs) {
    int i = 0; int out = 0; int p = 0;
    out = [];
    for (i = 0; i < len(xs); ++i) {
        p = 0;
        if (vector.vdot(model[0], xs[i]) + model[1] > 0.0) { p = 1; }
        out[i] = p;
    }
    return out;
}

# ---------------------------------------------------------------------------
# CART decision tree.
# model = [nodes, task, k]; each node is
#   [featureIdx, threshold, leftIdx, rightIdx, value, isLeaf]
# task: "gini" for classification (value = class), "mse" for regression
# (value = mean target). Built iteratively with an explicit stack.
# ---------------------------------------------------------------------------

fn tree_leaf_value(ys, idxs, task, k) {
    int counts = 0; int s = 0;
    if (task == "mse") {
        s = 0.0;
        for id in idxs { s = s + ys[id]; }
        return s / mathx.maxv(len(idxs), 1);
    }
    counts = arrayx.zerosf(k);
    for id in idxs { counts[ys[id]] = counts[ys[id]] + 1; }
    return arrayx.argmax(counts);
}

fn tree_impurity(ys, idxs, task, k) {
    int c = 0; int counts = 0; int g = 0; int m = 0; int n = 0; int p = 0; int s = 0;
    n = len(idxs);
    if (n == 0) { return 0.0; }
    if (task == "mse") {
        m = 0.0;
        for id in idxs { m = m + ys[id]; }
        m = m / n;
        s = 0.0;
        for id in idxs { s = s + (ys[id] - m) * (ys[id] - m); }
        return s / n;
    }
    counts = arrayx.zerosf(k);
    for id in idxs { counts[ys[id]] = counts[ys[id]] + 1; }
    g = 1.0;
    for (c = 0; c < k; ++c) {
        p = counts[c] / n;
        g = g - p * p;
    }
    return g;
}

# Best (feature, threshold) split by impurity decrease.
# Returns [feature, threshold, gain]; feature = -1 when nothing helps.
fn tree_best_split(xs, ys, idxs, task, k) {
    int a = 0; int b = 0; int bestF = 0; int bestGain = 0; int bestT = 0; int d = 0; int f = 0; int gain = 0; int imp = 0; int leftIdx = 0; int n = 0; int nl = 0; int nr = 0; int order = 0; int parentImp = 0; int rightIdx = 0; int t = 0; int thr = 0; int u = 0; int vals = 0;
    n = len(idxs);
    d = len(xs[0]);
    parentImp = tree_impurity(ys, idxs, task, k);
    bestGain = 1.0e-9;
    bestF = -1;
    bestT = 0.0;
    for (f = 0; f < d; ++f) {
        vals = [];
        for (t = 0; t < n; ++t) { vals[t] = xs[idxs[t]][f]; }
        order = arrayx.argsort(vals);
        for (t = 1; t < n; ++t) {
            a = vals[order[t - 1]];
            b = vals[order[t]];
            if (b - a < 1.0e-12) { continue; }
            thr = (a + b) * 0.5;
            leftIdx = [];
            rightIdx = [];
            for (u = 0; u < n; ++u) {
                if (vals[u] <= thr) { leftIdx[len(leftIdx)] = idxs[u]; }
                else { rightIdx[len(rightIdx)] = idxs[u]; }
            }
            nl = len(leftIdx);
            nr = len(rightIdx);
            if (nl == 0 || nr == 0) { continue; }
            imp = (nl * tree_impurity(ys, leftIdx, task, k) +
                   nr * tree_impurity(ys, rightIdx, task, k)) / n;
            gain = parentImp - imp;
            if (gain > bestGain) {
                bestGain = gain;
                bestF = f;
                bestT = thr;
            }
        }
    }
    return [bestF, bestT, bestGain];
}

# task: "gini" (classification; pass k = number of classes) or "mse"
# (regression; pass k = 0).
fn tree_fit(xs, ys, task, k, maxDepth, minSamples) {
    int popped = 0; int depth = 0; int f = 0; int i = 0; int idxs = 0; int leftIdx = 0; int li = 0; int makeLeaf = 0; int nodeIdx = 0; int nodes = 0; int ri = 0; int rightIdx = 0; int rootIdxs = 0; int split = 0; int stack = 0; int thr = 0; int u = 0;
    if (maxDepth == nil) { maxDepth = 6; }
    if (minSamples == nil) { minSamples = 2; }
    nodes = [];
    rootIdxs = [];
    for (i = 0; i < len(xs); ++i) { rootIdxs[i] = i; }
    nodes[0] = [-1, 0.0, -1, -1, 0.0, false];
    stack = [];
    stack[0] = [0, rootIdxs, 0];               # [nodeIdx, sampleIdxs, depth]
    while (len(stack) > 0) {
        popped = arrayx.pop(stack);
        stack = popped[0];
        top = popped[1];
        nodeIdx = top[0];
        idxs = top[1];
        depth = top[2];
        makeLeaf = false;
        if (depth >= maxDepth || len(idxs) < minSamples) { makeLeaf = true; }
        split = [-1, 0.0, 0.0];
        if (!makeLeaf) {
            split = tree_best_split(xs, ys, idxs, task, k);
            if (split[0] < 0) { makeLeaf = true; }
        }
        if (makeLeaf) {
            nodes[nodeIdx] = [-1, 0.0, -1, -1,
                              tree_leaf_value(ys, idxs, task, k), true];
            continue;
        }
        f = split[0];
        thr = split[1];
        leftIdx = [];
        rightIdx = [];
        for (u = 0; u < len(idxs); ++u) {
            if (xs[idxs[u]][f] <= thr) { leftIdx[len(leftIdx)] = idxs[u]; }
            else { rightIdx[len(rightIdx)] = idxs[u]; }
        }
        li = len(nodes);
        ri = li + 1;
        nodes[li] = [-1, 0.0, -1, -1, 0.0, false];
        nodes[ri] = [-1, 0.0, -1, -1, 0.0, false];
        nodes[nodeIdx] = [f, thr, li, ri, 0.0, false];
        stack[len(stack)] = [li, leftIdx, depth + 1];
        stack[len(stack)] = [ri, rightIdx, depth + 1];
    }
    return [nodes, task, k];
}

fn tree_predict_one(model, x) {
    int cur = 0; int node = 0; int nodes = 0;
    nodes = model[0];
    cur = 0;
    while (true) {
        node = nodes[cur];
        if (node[5]) { return node[4]; }
        if (x[node[0]] <= node[1]) { cur = node[2]; }
        else { cur = node[3]; }
    }
}

fn tree_predict(model, xs) {
    int i = 0; int out = 0;
    out = [];
    for (i = 0; i < len(xs); ++i) { out[i] = tree_predict_one(model, xs[i]); }
    return out;
}

fn tree_depth(model) {
    int popped = 0; int best = 0; int node = 0; int nodes = 0; int stack = 0;
    nodes = model[0];
    best = 0;
    stack = [];
    stack[0] = [0, 0];
    while (len(stack) > 0) {
        popped = arrayx.pop(stack);
        stack = popped[0];
        top = popped[1];
        node = nodes[top[0]];
        if (top[1] > best) { best = top[1]; }
        if (!node[5]) {
            stack[len(stack)] = [node[2], top[1] + 1];
            stack[len(stack)] = [node[3], top[1] + 1];
        }
    }
    return best;
}
