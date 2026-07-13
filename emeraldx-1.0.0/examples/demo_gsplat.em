# 3D gaussian splatting: EWA projection + front-to-back compositing.
import gsplat;
import mesh;
import quat;
import image;
import mathx;
import strx;
int a = 0;
int back = 0;
int cam = 0;
int cam2 = 0;
int col = 0;
int i = 0;
int ico = 0;
int img = 0;
int img2 = 0;
int ms = 0;
int p = 0;
int q = 0;
int splats = 0;
# emx-scope-safe

print("== emeraldx: gaussian splatting ==");
splats = [];
for (i = 0; i < 12; ++i) {
    a = mathx.TAU * i / 12.0;
    p = [cos(a) * 1.3, sin(a) * 1.3, sin(a * 2.0) * 0.3];
    col = [0.5 + 0.5 * cos(a), 0.5 + 0.5 * sin(a), 0.85];
    splats[len(splats)] = gsplat.make_splat(p, [0.26, 0.26, 0.26], quat.identity(), col, 0.85);
}
q = quat.from_axis_angle([0.0, 0.0, 1.0], 0.7);
splats[len(splats)] = gsplat.make_splat([0.0,0.0,0.0], [0.85, 0.16, 0.16], q, [1.0, 0.85, 0.2], 0.95);
print("hand-built scene  ", len(splats), " splats");

cam = gsplat.make_camera([0.0, -0.4, 4.6], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 46.0, 72, 54);
img = gsplat.render(splats, cam, [0.03, 0.03, 0.06]);
image.save(img, "examples/out/gsplat_ring.ppm");

# splatify a mesh: one oriented splat per face
ico = mesh.icosphere(1.0, 1);
ms = gsplat.from_mesh(ico, 0.9);
print("mesh -> splats    ", len(ms), " (one per face, normal-aligned)");
cam2 = gsplat.make_camera([2.4, 1.4, 2.8], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 45.0, 64, 48);
img2 = gsplat.render(ms, cam2, [0.02, 0.02, 0.04]);
image.save(img2, "examples/out/gsplat_sphere.ppm");

gsplat.save_ply(splats, "examples/out/splats.ply");
back = gsplat.load_ply("examples/out/splats.ply");
print("ply roundtrip     ", len(back), " splats");
print("wrote examples/out/gsplat_*.ppm, splats.ply");
image.print_ascii(image.resize(image.to_gray(img), 56, 18), nil);
