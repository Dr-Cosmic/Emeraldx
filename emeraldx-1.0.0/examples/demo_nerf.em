# A tiny NeRF trained from scratch: 3 posed views of a synthetic scene in,
# novel view out. Gradients flow analytically through the volume renderer.
# Recipe that converges fast: WHITE background + transparent init (sigmaBias).
import nerf;
import nn;
import prng;
import strx;
import image;
int allRd = 0;
int allRo = 0;
int allT = 0;
int bg = 0;
int ep = 0;
int eye = 0;
int eyeTest = 0;
int field = 0;
int fr = 0;
int gd = 0;
int gt = 0;
int gtImg = 0;
int h = 0;
int i = 0;
int idx = 0;
int lrImg = 0;
int opt = 0;
int r = 0;
int rays = 0;
int rd = 0;
int rds = 0;
int ro = 0;
int ros = 0;
int tg = 0;
int v = 0;
int views = 0;
int w = 0;
# emx-scope-safe

print("== emeraldx: neural radiance field ==");
w = 8; h = 8;
bg = [1.0, 1.0, 1.0];
views = [[2.6, 0.0, 0.0], [0.0, 0.5, 2.6], [-1.8, 0.4, -1.8]];
allRo = []; allRd = []; allT = [];
for (v = 0; v < 3; ++v) {
    eye = views[v];
    gt = nerf.scene_render_image(eye, [0.0,0.0,0.0], 40.0, w, h, 8, bg);
    rays = nerf.camera_rays(eye, [0.0,0.0,0.0], [0.0,1.0,0.0], 40.0, w, h);
    ros = rays[0]; rds = rays[1]; gd = gt[3];
    for (i = 0; i < w * h; ++i) {
        allRo[len(allRo)] = ros[i];
        allRd[len(allRd)] = rds[i];
        allT = allT + [gd[i*3], gd[i*3+1], gd[i*3+2]];
    }
}
print("training rays     ", len(allRo), " from ", len(views), " views");

fr = nerf.make_field(1, 16, prng.seed_from(12), -2.0);
field = fr[0];
opt = nn.opt_adam(field[0], 0.02);
for (ep = 0; ep < 21; ++ep) {
    v = ep % 3;
    ro = []; rd = []; tg = [];
    for (i = 0; i < w * h; ++i) {
        idx = v * w * h + i;
        ro[i] = allRo[idx];
        rd[i] = allRd[idx];
        tg = tg + [allT[idx*3], allT[idx*3+1], allT[idx*3+2]];
    }
    r = nerf.train_rays(field, opt, ro, rd, tg, 1.0, 4.2, 8, bg);
    field = r[0]; opt = r[1];
    if (ep % 5 == 0 || ep == 20) { print("  epoch ", ep, " loss ", strx.fmt_f(r[2], 5)); }
}

eyeTest = [1.8, 0.9, 1.8];
gtImg = nerf.scene_render_image(eyeTest, [0.0,0.0,0.0], 40.0, 10, 10, 8, bg);
lrImg = nerf.field_render_image(field, eyeTest, [0.0,0.0,0.0], 40.0, 10, 10, 8, bg);
print("held-out PSNR     ", strx.fmt_f(nerf.psnr(lrImg, gtImg), 2), " dB (novel view)");
image.save(gtImg, "examples/out/nerf_groundtruth.ppm");
image.save(lrImg, "examples/out/nerf_learned.ppm");
print("wrote examples/out/nerf_groundtruth.ppm, nerf_learned.ppm");
print("ground truth vs learned (novel view):");
image.print_ascii(gtImg, nil);
print("   ---");
image.print_ascii(lrImg, nil);
print("(more epochs sharpen it further; see docs/API.md#nerf)");
