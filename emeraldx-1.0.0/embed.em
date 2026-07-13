# =============================================================================
# embed.em - count-based word embeddings
#
# The classical pipeline that word2vec implicitly factorizes:
#   1. co-occurrence counts within a sliding window
#   2. PPMI (positive pointwise mutual information) reweighting
#   3. low-rank factorization (symmetric eigendecomposition) -> dense vectors
#
#   model = embed.fit(docsTokens, windowSize, dims)
#   v = embed.vector(model, "cat")
#   embed.similarity(model, "cat", "dog")
#   embed.nearest(model, "cat", 3)
#   embed.analogy(model, "king", "man", "woman", 3)   b - a + c
# =============================================================================

import mathx;
import arrayx;
import dictx;
import matrix;
import nlp;
# emx-scope-safe

# ---------------------------------------------------------------------------
# Co-occurrence within +-window (symmetric). Returns [vocab, counts] where
# vocab = [words, indexDict] and counts is a dense nWords x nWords matrix.
# ---------------------------------------------------------------------------

fn cooccurrence(docsTokens, window) {
    int counts = 0; int hi = 0; int i = 0; int idx = 0; int j = 0; int m = 0; int n = 0; int row = 0; int row2 = 0; int vocab = 0; int wi = 0; int wj = 0; int words = 0;
    if (window == nil) { window = 2; }
    vocab = nlp.build_vocab(docsTokens);
    words = vocab[0];
    idx = vocab[1];
    n = len(words);
    counts = matrix.zeros(n, n);
    for doc in docsTokens {
        m = len(doc);
        for (i = 0; i < m; ++i) {
            wi = dictx.dget(idx, doc[i], -1);
            if (wi < 0) { continue; }
            hi = mathx.mini(i + window, m - 1);
            for (j = i + 1; j <= hi; ++j) {
                wj = dictx.dget(idx, doc[j], -1);
                if (wj < 0) { continue; }
                row = counts[wi];
                row[wj] = row[wj] + 1.0;
                counts[wi] = row;
                row2 = counts[wj];
                row2[wi] = row2[wi] + 1.0;
                counts[wj] = row2;
            }
        }
    }
    return [vocab, counts];
}

# Positive pointwise mutual information of a co-occurrence matrix.
fn ppmi(counts) {
    int c = 0; int i = 0; int j = 0; int n = 0; int orow = 0; int out = 0; int ri = 0; int rj = 0; int row = 0; int rowSums = 0; int s = 0; int total = 0; int v = 0;
    n = len(counts);
    rowSums = [];
    total = 0.0;
    for (i = 0; i < n; ++i) {
        s = arrayx.sumv(counts[i]);
        rowSums[i] = s;
        total = total + s;
    }
    if (total < 1.0) { total = 1.0; }
    out = matrix.zeros(n, n);
    for (i = 0; i < n; ++i) {
        ri = rowSums[i];
        if (ri <= 0.0) { continue; }
        row = counts[i];
        orow = out[i];
        for (j = 0; j < n; ++j) {
            c = row[j];
            if (c <= 0.0) { continue; }
            rj = rowSums[j];
            v = ln((c * total) / (ri * rj));
            if (v > 0.0) { orow[j] = v; }
        }
        out[i] = orow;
    }
    return out;
}

# ---------------------------------------------------------------------------
# Fit: PPMI matrix -> top-k eigenpairs -> embeddings = V_k * sqrt(L_k).
# model = [words, indexDict, vectors(n x dims)]
# ---------------------------------------------------------------------------

fn fit(docsTokens, window, dims) {
    int co = 0; int eig = 0; int i = 0; int k = 0; int lams = 0; int lv = 0; int n = 0; int p = 0; int row = 0; int v = 0; int vecs = 0; int vocab = 0;
    if (dims == nil) { dims = 8; }
    co = cooccurrence(docsTokens, window);
    vocab = co[0];
    p = ppmi(co[1]);
    n = len(vocab[0]);
    if (dims > n) { dims = n; }
    eig = matrix.eig_jacobi(p, nil);
    lams = eig[0];
    v = eig[1];
    vecs = [];
    for (i = 0; i < n; ++i) {
        row = [];
        for (k = 0; k < dims; ++k) {
            lv = lams[k];
            if (lv < 0.0) { lv = 0.0; }
            row[k] = v[i][k] * sqrt(lv);
        }
        vecs[i] = row;
    }
    return [vocab[0], vocab[1], vecs];
}

fn vector(model, word) {
    int idx = 0; int vecs = 0;
    idx = model[1];
    if (dictx.dhas(idx, word) == false) { return nil; }
    vecs = model[2];
    return vecs[dictx.dget(idx, word, 0)];
}

fn cos(a, b) {
    int dot = 0; int i = 0; int na = 0; int nb = 0;
    dot = 0.0;
    na = 0.0;
    nb = 0.0;
    for (i = 0; i < len(a); ++i) {
        dot = dot + a[i] * b[i];
        na = na + a[i] * a[i];
        nb = nb + b[i] * b[i];
    }
    if (na < 1.0e-12 || nb < 1.0e-12) { return 0.0; }
    return dot / (sqrt(na) * sqrt(nb));
}

fn similarity(model, wa, wb) {
    int va = 0; int vb = 0;
    va = vector(model, wa);
    vb = vector(model, wb);
    if (va == nil || vb == nil) { return 0.0; }
    return cos(va, vb);
}

# k nearest words to `word` (excluding itself). Returns [[word, sim], ...].
fn nearest(model, word, k) {
    int target = 0;
    if (k == nil) { k = 3; }
    target = vector(model, word);
    if (target == nil) { return []; }
    return nearest_vec(model, target, k, word);
}

fn nearest_vec(model, target, k, excludeWord) {
    int i = 0; int j = 0; int n = 0; int order = 0; int out = 0; int sims = 0; int t = 0; int vecs = 0; int words = 0;
    words = model[0];
    vecs = model[2];
    sims = [];
    for (i = 0; i < len(words); ++i) {
        if (words[i] == excludeWord) { sims[i] = -2.0; }
        else { sims[i] = cos(target, vecs[i]); }
    }
    order = arrayx.argsort(sims);
    out = [];
    n = len(words);
    for (t = 0; t < k && t < n; ++t) {
        j = order[n - 1 - t];
        if (sims[j] <= -2.0) { continue; }
        out[len(out)] = [words[j], sims[j]];
    }
    return out;
}

# b - a + c  ("a is to b as c is to ?")
fn analogy(model, a, b, c, k) {
    int i = 0; int q = 0; int va = 0; int vb = 0; int vc = 0;
    if (k == nil) { k = 3; }
    va = vector(model, a);
    vb = vector(model, b);
    vc = vector(model, c);
    if (va == nil || vb == nil || vc == nil) { return []; }
    q = [];
    for (i = 0; i < len(va); ++i) { q[i] = vb[i] - va[i] + vc[i]; }
    return nearest_vec(model, q, k, c);
}
