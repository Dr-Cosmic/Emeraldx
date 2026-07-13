# =============================================================================
# ml_unsupervised.em - clustering and dimensionality reduction
#
#   kmeans_fit / kmeans_assign      k-means with k-means++ init
#   pca_fit / pca_transform         principal component analysis
#   dbscan                          density clustering (explicit queue)
#   agglomerative                   single-linkage hierarchical clustering
# =============================================================================

import mathx;
import arrayx;
import vector;
import matrix;
import mlcore;
import prng;
# emx-scope-safe

# ---------------------------------------------------------------------------
# k-means
# ---------------------------------------------------------------------------

# k-means++ initial centers. Returns [centers, newSeed].
fn kmeanspp_init(xs, k, seed) {
    int acc = 0; int best = 0; int c = 0; int cc = 0; int centers = 0; int d = 0; int dists = 0; int i = 0; int n = 0; int pick = 0; int r = 0; int r2 = 0; int target = 0; int total = 0;
    n = len(xs);
    centers = [];
    r = prng.randint(seed, 0, n - 1);
    seed = r[1];
    centers[0] = xs[r[0]];
    for (c = 1; c < k; ++c) {
        dists = [];
        total = 0.0;
        for (i = 0; i < n; ++i) {
            best = 1.0e30;
            for (cc = 0; cc < c; ++cc) {
                d = mlcore.sq_euclidean(xs[i], centers[cc]);
                if (d < best) { best = d; }
            }
            dists[i] = best;
            total = total + best;
        }
        r2 = prng.randf(seed);
        seed = r2[1];
        target = r2[0] * total;
        acc = 0.0;
        pick = n - 1;
        for (i = 0; i < n; ++i) {
            acc = acc + dists[i];
            if (acc >= target) { pick = i; break; }
        }
        centers[c] = xs[pick];
    }
    return [centers, seed];
}

fn kmeans_assign(xs, centers) {
    int bd = 0; int best = 0; int c = 0; int d = 0; int i = 0; int k = 0; int labels = 0; int n = 0;
    n = len(xs);
    k = len(centers);
    labels = [];
    for (i = 0; i < n; ++i) {
        best = 0;
        bd = mlcore.sq_euclidean(xs[i], centers[0]);
        for (c = 1; c < k; ++c) {
            d = mlcore.sq_euclidean(xs[i], centers[c]);
            if (d < bd) { bd = d; best = c; }
        }
        labels[i] = best;
    }
    return labels;
}

# Lloyd's algorithm. Returns [centers, labels, inertia, newSeed].
fn kmeans_fit(xs, k, maxIters, seed) {
    int c = 0; int centers = 0; int counts = 0; int d = 0; int i = 0; int inertia = 0; int init = 0; int it = 0; int j = 0; int labels = 0; int moved = 0; int n = 0; int newCenters = 0; int row = 0; int xi = 0;
    if (maxIters == nil) { maxIters = 50; }
    n = len(xs);
    d = len(xs[0]);
    init = kmeanspp_init(xs, k, seed);
    centers = init[0];
    seed = init[1];
    labels = [];
    for (it = 0; it < maxIters; ++it) {
        labels = kmeans_assign(xs, centers);
        newCenters = [];
        counts = [];
        for (c = 0; c < k; ++c) {
            newCenters[c] = arrayx.zerosf(d);
            counts[c] = 0;
        }
        for (i = 0; i < n; ++i) {
            c = labels[i];
            row = newCenters[c];
            xi = xs[i];
            for (j = 0; j < d; ++j) { row[j] = row[j] + xi[j]; }
            newCenters[c] = row;
            counts[c] = counts[c] + 1;
        }
        moved = 0.0;
        for (c = 0; c < k; ++c) {
            if (counts[c] > 0) {
                row = newCenters[c];
                for (j = 0; j < d; ++j) { row[j] = row[j] / counts[c]; }
                newCenters[c] = row;
            } else {
                newCenters[c] = centers[c];      # keep empty cluster in place
            }
            moved = moved + mlcore.sq_euclidean(newCenters[c], centers[c]);
        }
        centers = newCenters;
        if (moved < 1.0e-10) { break; }
    }
    labels = kmeans_assign(xs, centers);
    inertia = 0.0;
    for (i = 0; i < n; ++i) {
        inertia = inertia + mlcore.sq_euclidean(xs[i], centers[labels[i]]);
    }
    return [centers, labels, inertia, seed];
}

# ---------------------------------------------------------------------------
# PCA: model = [mean, components, explainedVar]
# components is d x nc; columns are principal directions.
# ---------------------------------------------------------------------------

fn pca_fit(xs, nComponents) {
    int a = 0; int b = 0; int c = 0; int ca = 0; int cent = 0; int comps = 0; int cov = 0; int cr = 0; int d = 0; int eig = 0; int expl = 0; int i = 0; int j = 0; int mu = 0; int n = 0; int row = 0; int vals = 0; int vecs = 0;
    n = len(xs);
    d = len(xs[0]);
    if (nComponents == nil) { nComponents = d; }
    mu = arrayx.zerosf(d);
    for (i = 0; i < n; ++i) {
        row = xs[i];
        for (j = 0; j < d; ++j) { mu[j] = mu[j] + row[j]; }
    }
    for (j = 0; j < d; ++j) { mu[j] = mu[j] / n; }
    cov = matrix.zeros(d, d);
    for (i = 0; i < n; ++i) {
        row = xs[i];
        cent = [];
        for (j = 0; j < d; ++j) { cent[j] = row[j] - mu[j]; }
        for (a = 0; a < d; ++a) {
            ca = cent[a];
            if (ca != 0.0) {
                cr = cov[a];
                for (b = 0; b < d; ++b) { cr[b] = cr[b] + ca * cent[b]; }
                cov[a] = cr;
            }
        }
    }
    cov = matrix.scale(cov, 1.0 / n);
    eig = matrix.eig_jacobi(cov, nil);
    vals = eig[0];
    vecs = eig[1];
    comps = matrix.submatrix(vecs, 0, d, 0, nComponents);
    expl = [];
    for (c = 0; c < nComponents; ++c) { expl[c] = mathx.maxv(vals[c], 0.0); }
    return [mu, comps, expl];
}

fn pca_transform(model, xs) {
    int cent = 0; int comps = 0; int i = 0; int j = 0; int mu = 0; int out = 0; int row = 0;
    mu = model[0];
    comps = model[1];
    out = [];
    for (i = 0; i < len(xs); ++i) {
        row = xs[i];
        cent = [];
        for (j = 0; j < len(row); ++j) { cent[j] = row[j] - mu[j]; }
        out[i] = matrix.vecmat(cent, comps);
    }
    return out;
}

# Reconstruct from PCA space back to the original space.
fn pca_inverse(model, zs) {
    int comps = 0; int ct = 0; int i = 0; int j = 0; int mu = 0; int out = 0; int row = 0;
    mu = model[0];
    comps = model[1];
    ct = matrix.transpose(comps);
    out = [];
    for (i = 0; i < len(zs); ++i) {
        row = matrix.vecmat(zs[i], ct);
        for (j = 0; j < len(row); ++j) { row[j] = row[j] + mu[j]; }
        out[i] = row;
    }
    return out;
}

# ---------------------------------------------------------------------------
# DBSCAN. Returns labels: -1 = noise, otherwise 0..(clusters-1).
# Uses an explicit frontier queue - no recursion.
# ---------------------------------------------------------------------------

fn dbscan(xs, epsRadius, minPts) {
    int cluster = 0; int eps2 = 0; int i = 0; int j = 0; int labels = 0; int n = 0; int nbrs = 0; int nbrs2 = 0; int q = 0; int qi = 0; int queue = 0;
    n = len(xs);
    eps2 = epsRadius * epsRadius;
    labels = [];
    for (i = 0; i < n; ++i) { labels[i] = -2; }         # -2 = unvisited
    cluster = 0;
    for (i = 0; i < n; ++i) {
        if (labels[i] != -2) { continue; }
        # find neighbours of i
        nbrs = [];
        for (j = 0; j < n; ++j) {
            if (mlcore.sq_euclidean(xs[i], xs[j]) <= eps2) {
                nbrs[len(nbrs)] = j;
            }
        }
        if (len(nbrs) < minPts) { labels[i] = -1; continue; }
        labels[i] = cluster;
        queue = nbrs;
        qi = 0;
        while (qi < len(queue)) {
            q = queue[qi];
            qi = qi + 1;
            if (labels[q] == -1) { labels[q] = cluster; }
            if (labels[q] != -2) { continue; }
            labels[q] = cluster;
            nbrs2 = [];
            for (j = 0; j < n; ++j) {
                if (mlcore.sq_euclidean(xs[q], xs[j]) <= eps2) {
                    nbrs2[len(nbrs2)] = j;
                }
            }
            if (len(nbrs2) >= minPts) {
                for v in nbrs2 { queue[len(queue)] = v; }
            }
        }
        cluster = cluster + 1;
    }
    return labels;
}

# ---------------------------------------------------------------------------
# Single-linkage agglomerative clustering down to k clusters.
# Returns labels 0..k-1. O(n^3) - use on small data.
# ---------------------------------------------------------------------------

fn agglomerative(xs, k) {
    int ba = 0; int bb = 0; int bestD = 0; int c = 0; int clusters = 0; int d = 0; int found = 0; int i = 0; int j = 0; int labels = 0; int n = 0; int out = 0; int remap = 0;
    n = len(xs);
    labels = [];
    for (i = 0; i < n; ++i) { labels[i] = i; }
    clusters = n;
    while (clusters > k) {
        bestD = 1.0e30;
        ba = -1;
        bb = -1;
        for (i = 0; i < n; ++i) {
            for (j = i + 1; j < n; ++j) {
                if (labels[i] != labels[j]) {
                    d = mlcore.sq_euclidean(xs[i], xs[j]);
                    if (d < bestD) {
                        bestD = d;
                        ba = labels[i];
                        bb = labels[j];
                    }
                }
            }
        }
        for (i = 0; i < n; ++i) {
            if (labels[i] == bb) { labels[i] = ba; }
        }
        clusters = clusters - 1;
    }
    # compact labels to 0..k-1
    remap = [];
    out = [];
    for (i = 0; i < n; ++i) {
        found = -1;
        for (c = 0; c < len(remap); ++c) {
            if (remap[c] == labels[i]) { found = c; break; }
        }
        if (found < 0) {
            found = len(remap);
            remap[len(remap)] = labels[i];
        }
        out[i] = found;
    }
    return out;
}
