# Computer vision pipeline: edges, threshold, morphology, components, corners.
import image;
import imgproc;
import strx;
int bin2 = 0;
int cc = 0;
int cc2 = 0;
int cl = 0;
int corners = 0;
int ed = 0;
int edges = 0;
int img = 0;
int n = 0;
int t = 0;
# emx-scope-safe

print("== emeraldx: computer vision ==");
img = image.new_image(48, 36, 1, 0.06);
img = image.fill_rect(img, 5, 5, 14, 14, [0.92]);
img = image.draw_circle(img, 33, 22, 8, [0.85]);
img = image.fill_rect(img, 30, 6, 6, 6, [0.5]);

t = imgproc.otsu_threshold(img);
print("otsu threshold    ", strx.fmt_f(t, 3));
bin2 = imgproc.threshold(img, t);
cc = imgproc.connected_components(bin2);
print("components        ", cc[1], " (three shapes)");

edges = imgproc.canny(img, 0.1, 0.3);
ed = edges[3];
n = 0;
for v in ed { if (v > 0.5) { n = n + 1; } }
print("canny edge px     ", n);
image.save(edges, "examples/out/cv_edges.pgm");

corners = imgproc.harris_corners(img, nil, 6);
print("harris corners    ", len(corners), " strongest at (",
      corners[0][0], ",", corners[0][1], ")");

cl = imgproc.closing(bin2, 1);
cc2 = imgproc.connected_components(cl);
print("after closing     ", cc2[1], " components");
image.save(img, "examples/out/cv_scene.pgm");
print("wrote examples/out/cv_scene.pgm, cv_edges.pgm");
image.print_ascii(image.resize(img, 48, 16), nil);
