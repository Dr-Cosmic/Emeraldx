# Software rendering: textured cube + shaded torus with z-buffering.
import mesh;
import uvmap;
import render;
import texture;
import image;
import mat4;
import strx;
int cam = 0;
int cube = 0;
int d = 0;
int dep = 0;
int i = 0;
int img = 0;
int lit = 0;
int proj = 0;
int target = 0;
int tex = 0;
int torus = 0;
int view = 0;
# emx-scope-safe

print("== emeraldx: software rasterizer ==");
cube = mesh.cube(1.4);
torus = mesh.torus(1.5, 0.45, 20, 10);
tex = uvmap.checker_test_pattern(32, 4);

cam = render.orbit_camera(4.6, 0.65, 0.45, 45.0, 80.0 / 60.0);
view = cam[0]; proj = cam[1];
target = render.make_target(80, 60, [0.04, 0.05, 0.09]);
target = render.draw_mesh(target, torus, mat4.rot_x(0.9), view, proj,
                          render.default_material([0.85, 0.35, 0.25]),
                          render.default_light());
target = render.draw_mesh(target, cube, mat4.translation(0.0, 0.35, 0.0), view, proj,
                          render.textured_material(tex), render.default_light());
render.save_target(target, "examples/out/render_scene.ppm");
img = target[0];
lit = 0;
d = img[3];
for (i = 0; i < 80 * 60; ++i) { if (d[i*3] + d[i*3+1] + d[i*3+2] > 0.3) { lit = lit + 1; } }
print("lit pixels        ", lit, " / 4800");
dep = render.depth_image(target);
image.save(dep, "examples/out/render_depth.pgm");
print("wrote examples/out/render_scene.ppm, render_depth.pgm");
image.print_ascii(image.resize(image.to_gray(img), 64, 20), nil);
