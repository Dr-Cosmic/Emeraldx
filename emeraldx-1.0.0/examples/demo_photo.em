# Computational photography: HDR merge, tonemap, white balance, denoise.
import image;
import photo;
import texture;
import prng;
import strx;
import arrayx;
int cast = 0;
int dem = 0;
int den = 0;
int e0 = 0;
int e1 = 0;
int e2 = 0;
int g = 0;
int gr = 0;
int hdr = 0;
int ldr = 0;
int mos = 0;
int noisy = 0;
int scene = 0;
int wb = 0;
# emx-scope-safe

print("== emeraldx: computational photography ==");
scene = texture.gradient_tex(48, 32, [0.15, 0.25, 0.7], [1.4, 1.1, 0.5]);
scene = image.fill_rect(scene, 6, 8, 12, 12, [1.8, 0.4, 0.25]);

e0 = image.clamp_image(photo.exposure(scene, -2.0));
e1 = image.clamp_image(scene);
e2 = image.clamp_image(photo.exposure(scene, 2.0));
hdr = photo.merge_exposures([e0, e1, e2], [-2.0, 0.0, 2.0]);
print("hdr max radiance  ", strx.fmt_f(arrayx.maxv(hdr[3]), 2));
ldr = photo.reinhard_tonemap(hdr, 0.35, 3.0);
print("tonemapped max    ", strx.fmt_f(arrayx.maxv(ldr[3]), 3));
image.save(image.clamp_image(ldr), "examples/out/photo_tonemapped.ppm");

cast = photo.scale_channels(e1, [0.75, 0.95, 1.25]);
wb = photo.gray_world_wb(cast);
image.save(image.clamp_image(wb), "examples/out/photo_whitebalanced.ppm");
print("white balance     done (gray-world)");

g = image.to_gray(e1);
gr = photo.film_grain(g, 0.12, prng.seed_from(4));
noisy = gr[0];
den = photo.bilateral_filter(noisy, 2, nil, nil);
fn mad(a, b) {
    int da = 0; int db = 0; int dv = 0; int i = 0; int s = 0;
    int s = 0; int i = 0; int dv = 0; int da = 0; int db = 0;
    s = 0.0; da = a[3]; db = b[3];
    for (i = 0; i < len(da); ++i) {
        dv = da[i] - db[i];
        if (dv < 0.0) { dv = 0.0 - dv; }
        s = s + dv;
    }
    return s / len(da);
}
print("noise MAD         ", strx.fmt_f(mad(noisy, g), 4), " -> bilateral ", strx.fmt_f(mad(den, g), 4));

mos = photo.bayer_mosaic(e1);
dem = photo.demosaic_bilinear(mos);
print("demosaic MAD      ", strx.fmt_f(mad(dem, e1), 4));
print("wrote examples/out/photo_*.ppm");
