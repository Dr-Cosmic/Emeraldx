# Classical ML: regression, classification, clustering, PCA.
import prng;
import dataset;
import ml_supervised;
import ml_unsupervised;
import mlcore;
import strx;
int c0 = 0;
int c1 = 0;
int centers = 0;
int comps = 0;
int cx = 0;
int cy = 0;
int dir0 = 0;
int i = 0;
int km = 0;
int kn = 0;
int lbl = 0;
int lg = 0;
int model = 0;
int off = 0;
int pc = 0;
int plg = 0;
int pred = 0;
int pt = 0;
int r0 = 0;
int r1 = 0;
int rn = 0;
int sd = 0;
int sp = 0;
int tr = 0;
int xs = 0;
int xte = 0;
int xtr = 0;
int ys = 0;
int yte = 0;
int ytr = 0;
# emx-scope-safe

print("== emeraldx: machine learning ==");
# --- regression: y = 3 x0 - 2 x1 + 1 + noise
sd = prng.seed_from(7);
xs = []; ys = [];
for (i = 0; i < 60; ++i) {
    r0 = prng.uniform(sd, -2.0, 2.0); sd = r0[1];
    r1 = prng.uniform(sd, -2.0, 2.0); sd = r1[1];
    rn = prng.normal(sd, 0.0, 0.05); sd = rn[1];
    xs[i] = [r0[0], r1[0]];
    ys[i] = 3.0 * r0[0] - 2.0 * r1[0] + 1.0 + rn[0];
}
model = ml_supervised.linreg_fit(xs, ys);
pred = ml_supervised.linreg_predict(model, xs);
print("linreg R2         ", strx.fmt_f(mlcore.r2_score(pred, ys), 4));
print("linreg weights    ", strx.fmt_arr(model[0], 3, ", "), " bias ", strx.fmt_f(model[1], 3));

# --- classification: two gaussian blobs
cx = []; cy = [];
for (i = 0; i < 40; ++i) {
    r0 = prng.normal(sd, 0.0, 0.6); sd = r0[1];
    r1 = prng.normal(sd, 0.0, 0.6); sd = r1[1];
    off = 0.0;
    lbl = 0;
    if (i % 2 == 1) { off = 2.5; lbl = 1; }
    cx[i] = [r0[0] + off, r1[0] + off];
    cy[i] = lbl;
}
sp = dataset.train_test_split(cx, cy, 0.25, prng.seed_from(3));
xtr = sp[0]; ytr = sp[1]; xte = sp[2]; yte = sp[3];
lg = ml_supervised.logistic_fit(xtr, ytr, 0.3, 300, 0.0);
plg = ml_supervised.logistic_predict(lg, xte, 0.5);
print("logistic acc      ", strx.fmt_f(mlcore.accuracy(plg, yte), 3));
kn = ml_supervised.knn_predict(ml_supervised.knn_model(xtr, ytr, 3), xte);
print("knn(3) acc        ", strx.fmt_f(mlcore.accuracy(kn, yte), 3));
tr = ml_supervised.tree_fit(xtr, ytr, "classify", 2, 4, 2);
pt = ml_supervised.tree_predict(tr, xte);
print("tree acc          ", strx.fmt_f(mlcore.accuracy(pt, yte), 3));

# --- clustering + PCA
km = ml_unsupervised.kmeans_fit(cx, 2, 25, prng.seed_from(5));
centers = km[0];
c0 = centers[0]; c1 = centers[1];
print("kmeans centers    (", strx.fmt_f(c0[0],2), ",", strx.fmt_f(c0[1],2), ") (",
      strx.fmt_f(c1[0],2), ",", strx.fmt_f(c1[1],2), ")");
pc = ml_unsupervised.pca_fit(cx, 1);
comps = pc[1];
print("pca direction     (", strx.fmt_f(comps[0][0],3), ",", strx.fmt_f(comps[1][0],3), ")  (~diagonal 0.707)");
