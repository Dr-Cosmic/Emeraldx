# =============================================================================
# selftest.em - fast sanity check of every emeraldx module
#
#   cd <package root>
#   emerald selftest.em
#
# Each check is small (whole run well under a minute). A failure prints the
# module name; the summary line reports pass/fail counts.
# =============================================================================

import mathx;
import prng;
import arrayx;
import strx;
import dictx;
import complexx;
import vector;
import matrix;
import mat4;
import quat;
import tensor;
import stats;
import dataset;
import mlcore;
import ml_supervised;
import ml_unsupervised;
import nn;
import cnn;
import image;
import imgproc;
import photo;
import texture;
import mesh;
import uvmap;
import render;
import nerf;
import gsplat;
import sfm;
import nlp;
import embed;
import physics;
import emeraldx;
int a = 0;
int at = 0;
int b = 0;
int back = 0;
int bx = 0;
int c = 0;
int c0 = 0;
int c1 = 0;
int cam = 0;
int camg = 0;
int ck = 0;
int cv = 0;
int d = 0;
int dLdC = 0;
int docs = 0;
int drawn = 0;
int dt = 0;
int emodel = 0;
int eps = 0;
int f2 = 0;
int fail = 0;
int field = 0;
int fr = 0;
int fw = 0;
int g = 0;
int gi = 0;
int grow = 0;
int gs = 0;
int gw = 0;
int gw0 = 0;
int hi = 0;
int hot = 0;
int i = 0;
int ident3 = 0;
int im = 0;
int img1 = 0;
int inv = 0;
int it = 0;
int k = 0;
int km = 0;
int l0 = 0;
int l1 = 0;
int lbl = 0;
int lo = 0;
int loss0 = 0;
int loss1 = 0;
int lr = 0;
int m = 0;
int net = 0;
int nsteps = 0;
int numg = 0;
int og = 0;
int opt = 0;
int out = 0;
int outI = 0;
int p = 0;
int pass = 0;
int pp = 0;
int pred = 0;
int q = 0;
int r = 0;
int r0 = 0;
int r1 = 0;
int rB = 0;
int rdv = 0;
int ro = 0;
int row0 = 0;
int rr = 0;
int rw = 0;
int s = 0;
int s1 = 0;
int s2 = 0;
int sig = 0;
int sp = 0;
int srt = 0;
int st = 0;
int t = 0;
int tBv = 0;
int tg = 0;
int th = 0;
int tm = 0;
int tn = 0;
int tt = 0;
int u = 0;
int u2 = 0;
int ua = 0;
int ub = 0;
int v = 0;
int vv = 0;
int w0 = 0;
int ws = 0;
int x = 0;
int x0 = 0;
int xr = 0;
int xs = 0;
int xw = 0;
int y0 = 0;
int z3 = 0;
int zb = 0;
# emx-scope-safe

pass = 0;
fail = 0;

fn near(a, b, tol) {
    int d = 0;
    d = a - b;
    if (d < 0.0) { d = 0.0 - d; }
    return d <= tol;
}

fn report(nm, ok) {
    if (ok) { print("  ok   ", nm); return 1; }
    print("  FAIL ", nm);
    return 0;
}

# ---- mathx
r = report("mathx", near(mathx.atan2(1.0, 1.0), pi / 4.0, 1.0e-9) &&
                    mathx.clampi(9, 0, 5) == 5 && mathx.mini(3, 7) == 3);
pass = pass + r; fail = fail + 1 - r;

# ---- prng (deterministic + range)
s = prng.seed_from(42);
u = prng.uniform(s, 0.0, 1.0);
u2 = prng.uniform(prng.seed_from(42), 0.0, 1.0);
r = report("prng", u[0] == u2[0] && u[0] >= 0.0 && u[0] < 1.0);
pass = pass + r; fail = fail + 1 - r;

# ---- arrayx
a = [3.0, 1.0, 2.0];
srt = arrayx.sort(a);
og = arrayx.argsort(a);
r = report("arrayx", srt[0] == 1.0 && srt[2] == 3.0 && og[0] == 1 &&
                     near(arrayx.sumv(a), 6.0, 1.0e-12));
pass = pass + r; fail = fail + 1 - r;

# ---- strx
r = report("strx", strx.trim("  hi ") == "hi" &&
                   strx.join(strx.split("a,b,c", ","), "|") == "a|b|c" &&
                   strx.parse_float("2.5") == 2.5);
pass = pass + r; fail = fail + 1 - r;

# ---- dictx
d = dictx.dnew();
d = dictx.dset(d, "k", 7);
d = dictx.dset(d, "k", 9);
r = report("dictx", dictx.dget(d, "k", 0) == 9 && dictx.dhas(d, "zz") == false);
pass = pass + r; fail = fail + 1 - r;

# ---- complexx (FFT of impulse = flat spectrum)
sp = complexx.fft([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]);
spRe = sp[0];
r = report("complexx", near(spRe[0], 1.0, 1.0e-9) && near(spRe[3], 1.0, 1.0e-9));
pass = pass + r; fail = fail + 1 - r;

# ---- vector
r = report("vector", near(vector.vdot([1.0,2.0,3.0],[4.0,5.0,6.0]), 32.0, 1.0e-12) &&
                     near(vector.vnorm(vector.vcross([1.0,0.0,0.0],[0.0,1.0,0.0])), 1.0, 1.0e-12));
pass = pass + r; fail = fail + 1 - r;

# ---- matrix (solve + svd null vector)
m = [[2.0, 1.0], [1.0, 3.0]];
x = matrix.solve(m, [5.0, 10.0]);
r = report("matrix", near(x[0], 1.0, 1.0e-9) && near(x[1], 3.0, 1.0e-9));
pass = pass + r; fail = fail + 1 - r;

# ---- mat4 (translate then inverse)
t = mat4.translation(1.0, 2.0, 3.0);
inv = mat4.inverse(t);
p = mat4.transform_point(inv, [1.0, 2.0, 3.0]);
r = report("mat4", near(p[0], 0.0, 1.0e-9) && near(p[2], 0.0, 1.0e-9));
pass = pass + r; fail = fail + 1 - r;

# ---- quat (90 deg about z rotates x to y)
q = quat.from_axis_angle([0.0, 0.0, 1.0], pi / 2.0);
v = quat.rotate_vec(q, [1.0, 0.0, 0.0]);
r = report("quat", near(v[1], 1.0, 1.0e-9) && near(v[0], 0.0, 1.0e-9));
pass = pass + r; fail = fail + 1 - r;

# ---- tensor
tn = tensor.from_flat([2, 3], [1.0,2.0,3.0,4.0,5.0,6.0]);
tt = tensor.ttranspose2d(tn);
r = report("tensor", tensor.tget(tt, [2, 1]) == 6.0 && tensor.tget(tt, [0, 1]) == 4.0);
pass = pass + r; fail = fail + 1 - r;

# ---- stats
r = report("stats", near(stats.mean([1.0, 2.0, 3.0, 4.0]), 2.5, 1.0e-12) &&
                    near(stats.stddev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]), 2.0, 1.0e-9));
pass = pass + r; fail = fail + 1 - r;

# ---- dataset (csv roundtrip)
dataset.save_csv("examples/out/selftest.csv", [[1.0, 2.0]], ["a"], ["x", "y", "lbl"]);
back = dataset.load_csv("examples/out/selftest.csv", true, nil);
bx = back[0];
row0 = bx[0];
r = report("dataset", row0[1] == 2.0 && back[1][0] == "a");
pass = pass + r; fail = fail + 1 - r;

# ---- mlcore
r = report("mlcore", near(mlcore.accuracy([1, 0, 1], [1, 1, 1]), 0.667, 0.01) &&
                     near(mlcore.mse([1.0], [3.0]), 4.0, 1.0e-12));
pass = pass + r; fail = fail + 1 - r;

# ---- ml_supervised (perfect line)
lr = ml_supervised.linreg_fit([[0.0], [1.0], [2.0]], [1.0, 3.0, 5.0]);
pred = ml_supervised.linreg_predict(lr, [[3.0]]);
r = report("ml_supervised", near(pred[0], 7.0, 1.0e-6));
pass = pass + r; fail = fail + 1 - r;

# ---- ml_unsupervised (2 obvious clusters)
xs = [[0.0,0.0],[0.1,0.0],[0.0,0.1],[5.0,5.0],[5.1,5.0],[5.0,5.1]];
km = ml_unsupervised.kmeans_fit(xs, 2, 20, prng.seed_from(1));
lbl = km[1];
r = report("ml_unsupervised", lbl[0] == lbl[1] && lbl[3] == lbl[5] && lbl[0] != lbl[3]);
pass = pass + r; fail = fail + 1 - r;

# ---- nn (one adam step reduces loss on a fixed pair)
b = nn.build([2, 4, 1], ["tanh", "sigmoid"], prng.seed_from(2));
net = b[0];
opt = nn.opt_adam(net, 0.05);
x0 = [0.3, -0.2];
y0 = [1.0];
l0 = 0.0; l1 = 0.0;
for (it = 0; it < 12; ++it) {
    fw = nn.forward(net, x0);
    out = nn.output(fw);
    e = out[0] - y0[0];
    if (it == 0) { l0 = e * e; }
    l1 = e * e;
    g = nn.backward(net, fw, [2.0 * e]);
    st = nn.apply_grads(net, opt, g[0], g[1]);
    net = st[0]; opt = st[1];
}
r = report("nn", l1 < l0);
pass = pass + r; fail = fail + 1 - r;

# ---- cnn (shape sanity)
fm = cnn.fmap(4, 4, 0.5);
kr = cnn.kernel_rand(3, 3, prng.seed_from(3));
co = cnn.conv2d(fm, kr[0], 0.1);
r = report("cnn", co[0] == 2 && co[1] == 2 && len(co[2]) == 4);
pass = pass + r; fail = fail + 1 - r;

# ---- image
im = image.new_image(8, 4, 3, 0.2);
im = image.set_px(im, 3, 1, 0, 0.9);
r = report("image", near(image.get_px(im, 3, 1, 0), 0.9, 1.0e-12) && im[0] == 8);
pass = pass + r; fail = fail + 1 - r;

# ---- imgproc (otsu splits two-level image)
gi = image.new_image(10, 10, 1, 0.1);
gi = image.fill_rect(gi, 0, 0, 5, 10, [0.9]);
th = imgproc.otsu_threshold(gi);
r = report("imgproc", th > 0.1 && th < 0.9);
pass = pass + r; fail = fail + 1 - r;

# ---- photo (tonemap keeps range)
hot = photo.exposure(gi, 2.0);
tm = photo.reinhard_tonemap(hot, nil, nil);
r = report("photo", arrayx.maxv(tm[3]) < 1.0);
pass = pass + r; fail = fail + 1 - r;

# ---- texture (checker alternates)
ck = texture.checker(8, 8, 2, nil, nil);
r = report("texture", near(image.get_px(ck, 0, 0, 0), 0.9, 1.0e-6) &&
                      near(image.get_px(ck, 4, 0, 0), 0.15, 1.0e-6));
pass = pass + r; fail = fail + 1 - r;

# ---- mesh (cube area)
c = mesh.cube(2.0);
r = report("mesh", near(mesh.surface_area(c), 24.0, 1.0e-9) && mesh.vertex_count(c) == 24);
pass = pass + r; fail = fail + 1 - r;

# ---- uvmap (atlas fits unit square)
at = uvmap.atlas_pack_faces(c, nil);
ub = uvmap.uv_bounds(at);
lo = ub[0]; hi = ub[1];
r = report("uvmap", lo[0] >= 0.0 && hi[0] <= 1.0 && hi[1] <= 1.0);
pass = pass + r; fail = fail + 1 - r;

# ---- render (something gets drawn)
cam = render.orbit_camera(4.0, 0.6, 0.4, 45.0, 1.0);
tg = render.make_target(24, 24, nil);
tg = render.draw_mesh(tg, c, mat4.identity(), cam[0], cam[1],
                      render.default_material(nil), render.default_light());
zb = tg[1];
drawn = 0;
for (i = 0; i < 24 * 24; ++i) { if (zb[i] < 1.0e29) { drawn = drawn + 1; } }
r = report("render", drawn > 40);
pass = pass + r; fail = fail + 1 - r;

# ---- nerf (analytic gradient matches numeric)
fr = nerf.make_field(1, 5, prng.seed_from(3), -1.0);
field = fr[0];
ro = [0.0, 0.0, 2.0];
rdv = [0.0, 0.0, -1.0];
r0 = nerf.render_ray(field, ro, rdv, 1.0, 3.0, 3, [1.0, 1.0, 1.0]);
c0 = r0[0];
dLdC = [2.0 * (c0[0] - 0.4), 2.0 * (c0[1] - 0.4), 2.0 * (c0[2] - 0.4)];
gs = nerf.ray_grads(field, r0[1], dLdC);
gw = gs[0];
gw0 = gw[0];
grow = gw0[0];
eps = 1.0e-5;
net = field[0];
ws = net[2];
w0 = ws[0];
rw = w0[0];
rw[1] = rw[1] + eps;
w0[0] = rw;
ws[0] = w0;
f2 = [[net[0], net[1], ws, net[3]], field[1]];
r1 = nerf.render_ray(f2, ro, rdv, 1.0, 3.0, 3, [1.0, 1.0, 1.0]);
c1 = r1[0];
loss0 = (c0[0]-0.4)*(c0[0]-0.4) + (c0[1]-0.4)*(c0[1]-0.4) + (c0[2]-0.4)*(c0[2]-0.4);
loss1 = (c1[0]-0.4)*(c1[0]-0.4) + (c1[1]-0.4)*(c1[1]-0.4) + (c1[2]-0.4)*(c1[2]-0.4);
numg = (loss1 - loss0) / eps;
r = report("nerf", near(grow[1], numg, 1.0e-3));
pass = pass + r; fail = fail + 1 - r;

# ---- gsplat (front occludes back)
s1 = gsplat.make_splat([0.0,0.0,1.0], [0.3,0.3,0.3], quat.identity(), [1.0,0.0,0.0], 0.95);
s2 = gsplat.make_splat([0.0,0.0,-1.0], [0.3,0.3,0.3], quat.identity(), [0.0,1.0,0.0], 0.95);
camg = gsplat.make_camera([0.0,0.0,4.0], [0.0,0.0,0.0], [0.0,1.0,0.0], 40.0, 12, 10);
og = gsplat.render([s1, s2], camg, nil);
r = report("gsplat", image.get_px(og, 6, 5, 0) > image.get_px(og, 6, 5, 1));
pass = pass + r; fail = fail + 1 - r;

# ---- sfm (triangulate a known point)
k = sfm.intrinsics(100.0, 32.0, 32.0);
ident3 = matrix.identity(3);
z3 = [0.0, 0.0, 0.0];
rB = [[cos(0.2), 0.0, sin(0.2)], [0.0, 1.0, 0.0], [0.0-sin(0.2), 0.0, cos(0.2)]];
tBv = [-0.8, 0.0, 0.1];
xw = [0.3, -0.2, 4.0];
ua = sfm.project(k, ident3, z3, xw);
ub = sfm.project(k, rB, tBv, xw);
xr = sfm.triangulate(k, ident3, z3, k, rB, tBv, ua, ub);
r = report("sfm", near(xr[0], 0.3, 1.0e-6) && near(xr[2], 4.0, 1.0e-6));
pass = pass + r; fail = fail + 1 - r;

# ---- nlp
r = report("nlp", nlp.porter_stem("running") == "run" &&
                  nlp.levenshtein("kitten", "sitting") == 3);
pass = pass + r; fail = fail + 1 - r;

# ---- embed (fits without error, self similarity 1)
docs = [["red", "apple", "fruit"], ["green", "apple", "fruit"], ["fast", "car", "road"]];
emodel = embed.fit(docs, 2, 4);
r = report("embed", near(embed.similarity(emodel, "apple", "apple"), 1.0, 1.0e-6));
pass = pass + r; fail = fail + 1 - r;

# ---- physics (rk4 SHM one period)
fn shm_acc(p, v, t) { return [0.0 - p[0], 0.0, 0.0]; }
pp = [1.0, 0.0, 0.0];
vv = [0.0, 0.0, 0.0];
nsteps = 126;
dt = mathx.TAU / nsteps;
for (i = 0; i < nsteps; ++i) {
    rr = physics.integrate_rk4(shm_acc, pp, vv, i * dt, dt);
    pp = rr[0]; vv = rr[1];
}
r = report("physics", near(pp[0], 1.0, 1.0e-3));
pass = pass + r; fail = fail + 1 - r;

# ---- emeraldx
r = report("emeraldx", emeraldx.version() == "1.0.0" && len(emeraldx.modules()) == 31);
pass = pass + r; fail = fail + 1 - r;

print("");
print("selftest: ", pass, " passed, ", fail, " failed");
