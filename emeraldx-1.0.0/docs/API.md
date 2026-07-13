# emeraldx API reference

Formats and conventions are described in `docs/DESIGN.md`. Optional trailing
parameters may be omitted entirely (missing arguments arrive as `nil` and
functions substitute documented defaults).

## mathx

Scalar helpers shared by everything else. Constants (`TAU`, `E`, `GOLDEN`), clamps, `atan2`, `hypot`, degree/radian conversion, `logistic`, `erf`, `lgamma`/`tgamma` approximations, root finding (`bisect` takes a function value), and integer `mini`/`maxi`.

```
maxv(a, b)
minv(a, b)
max3(a, b, c)
min3(a, b, c)
clampf(v, lo, hi)
mini(a, b)
maxi(a, b)
clampi(v, lo, hi)
sign(v)
roundi(x)
roundto(x, decimals)
fmod(a, b)
lerp(a, b, t)
unlerp(a, b, v)
map_range(v, a0, a1, b0, b1)
smootherstep(a, b, t)
deg(radians)
rad(degrees)
wrap_angle(a)
atan2(y, x)
hypot(x, y)
hypot3(x, y, z)
sqr(x)
cube(x)
cbrt(x)
powi(base, n)
isclose(a, b, tol)
is_nan(x)
gcd(a, b)
lcm(a, b)
factorial(n)
ncr(n, k)
npr(n, k)
sinc(x)
erf(x)
erfc(x)
gammafn(x)
ln_safe(x)
softplus(x)
logistic(x)
bisect(f, lo, hi, iters)
deriv(f, x, h)
integrate(f, a, b, n)
```

## prng

Deterministic pseudo-randomness built on a functional MINSTD core: every call takes a state and returns `[value, newState]`. `uniform`, `normal` (Box-Muller), `randint`, `normal_array`, `shuffle`, plus a stateful `Rng` class whose internal field is deliberately named `emxState` (see DESIGN.md on scoping).

```
seed_from(x)
lcg(s)
randf(s)
uniform(s, lo, hi)
randint(s, lo, hi)
chance(s, p)
normal(s)
gauss(s, mu, sigma)
randf_array(s, n, lo, hi)
normal_array(s, n)
choice(s, arr)
weighted_index(s, weights)
```

## arrayx

Flat-array utilities: constructors (`zerosf`, `full`, `linspace`), reductions (`sumv`, `minv`, `argmax`), quicksort `sort`/`argsort`, `unique`, `shuffle`, `sample`, and functional `map`/`filter`/`reduce` taking function values. `pop(a)` returns `[rest, value]` because arrays are values.

```
zeros(n)
zerosf(n)
full(n, v)
range(start, stop, step)
linspace(a, b, n)
copy(a)
push(a, v)
pop(a)
slice(a, i, j)
take(a, n)
drop(a, n)
reverse(a)
insert_at(a, idx, v)
remove_at(a, idx)
swap(a, i, j)
contains(a, v)
index_of(a, v)
count_of(a, v)
sumv(a)
prodv(a)
minv(a)
maxv(a)
argmin(a)
argmax(a)
flatten1(a)
repeat(a, times)
arr_eq(a, b)
map(a, f)
filter(a, pred)
reduce(a, f, init)
any_of(a, pred)
all_of(a, pred)
sort(a)
sort_desc(a)
argsort(vals)
gather(a, order)
unique(a)
shuffle(a, seed)
sample(a, k, seed)
```

## strx

String toolkit: case mapping, trim, split (`split`, `split_ws`, `split_lines`), join, search, `replace_all`, padding, parsing (`parse_int`, `parse_float`), and numeric formatting (`fmt_f`, `fmt_arr`, `fmt_g`).

```
is_digit(c)
is_lower(c)
is_upper(c)
is_alpha(c)
is_alnum(c)
is_space(c)
is_vowel(c)
substr(s, start, count)
char_str(c)
to_upper(s)
to_lower(s)
capitalize(s)
ltrim(s)
rtrim(s)
trim(s)
starts_with(s, prefix)
ends_with(s, suffix)
index_of(s, sub, fromIdx)
contains(s, sub)
count_of(s, sub)
replace_all(s, fromS, toS)
repeat(s, times)
pad_left(s, width, padChar)
pad_right(s, width, padChar)
reverse(s)
split(s, delim)
split_ws(s)
split_lines(s)
join(arr, sep)
parse_int(s)
parse_float(s)
is_number(s)
fmt_f(x, decimals)
fmt_arr(arr, decimals, sep)
fmt_g(x)
```

## dictx

A string-keyed dictionary as a value type: two sorted parallel arrays with binary search. `dnew`, `dset` (returns the updated dict), `dget` with default, `dhas`, `ddel`, `dkeys`, `dvalues`, `dmerge`.

```
dnew()
dsize(d)
dfind(d, k)
dhas(d, k)
dget(d, k, default)
dset(d, k, v)
dinc(d, k, by)
ddel(d, k)
dkeys(d)
dvals(d)
dargmax(d)
dsorted_by_value_desc(d)
```

## complexx

Complex arithmetic on `[re, im]` pairs and a radix-2 iterative FFT working on separate real/imag arrays: `fft(re, im)`, `ifft`, `fft_magnitudes`, plus `cmul`, `cexp`, `cpolar`.

```
cnum(re, im)
cadd(a, b)
csub(a, b)
cmul(a, b)
cdiv(a, b)
cconj(a)
cabs(a)
carg(a)
cscale(a, s)
from_polar(r, theta)
cexp(a)
cpow(a, p)
csqrt(a)
next_pow2(n)
pad_to(arr, n, fill)
bit_reverse(i, bits)
fft(reIn, imIn)
ifft(reIn, imIn)
spectrum(signal)
convolve_fft(a, b)
convolve_direct(a, b)
dft_naive(signal)
hann_window(n)
hamming_window(n)
apply_window(signal, w)
sine_wave(n, cycles, amplitude, phase)
```

## vector

N-dimensional vectors as flat arrays: `vadd`/`vsub`/`vscale`/`vdot`, norms and distances, `vnormalize`, `vlerp`, and 3D-specific `vcross`, `vreflect`, `vproject`.

```
vec2(x, y)
vec3(x, y, z)
vec4(x, y, z, w)
vzeros(n)
vfull(n, x)
vadd(a, b)
vsub(a, b)
vscale(a, s)
vneg(a)
vhadamard(a, b)
vaxpy(a, b, s)
vdot(a, b)
vnorm_sq(a)
vnorm(a)
vdist(a, b)
vdist_sq(a, b)
vnormalize(a)
vlerp(a, b, t)
vcross(a, b)
vangle(a, b)
vproject(a, b)
vreject(a, b)
vreflect(v, n)
vclamp_norm(v, maxLen)
vabs(a)
vmin(a, b)
vmax(a, b)
vsum(a)
vmean(a)
vperp3(a)
vslerp(a, b, t)
```

## matrix

Dense matrices as nested row arrays. Construction (`zeros`, `identity`, `diag`, `from_flat`), arithmetic, `matmul`/`matvec`, `transpose`, `submatrix`. Decompositions: `lu`, `det`, `solve`, `qr` (Householder), `lstsq`, symmetric `eig_jacobi`, general `svd` via A^T A (with orthonormal completion of rank-deficient U columns and a relative rank cutoff), `null_vector`, `pinv`, `rank`.

```
rows(m)
cols(m)
mnew(r, c, fill)
zeros(r, c)
ones(r, c)
identity(n)
diag(v)
get_diag(m)
from_flat(flat, r, c)
to_flat(m)
rand(r, c, seed, lo, hi)
randn(r, c, seed, scale)
add(a, b)
sub(a, b)
hadamard(a, b)
scale(a, s)
add_scalar(a, s)
apply(a, f)
matmul(a, b)
matvec(a, v)
vecmat(v, a)
outer(u, v)
transpose(a)
trace(a)
row(m, i)
col(m, j)
set_row(m, i, v)
set_col(m, j, v)
swap_rows(m, i, j)
hcat(a, b)
vcat(a, b)
submatrix(m, r0, r1, c0, c1)
norm_fro(a)
norm_max(a)
col_means(a)
print_mat(m, decimals)
lu(aIn)
det(aIn)
lu_solve(luRes, b)
solve(a, b)
inverse(a)
qr(a)
lstsq(a, b)
eig_power(a, iters, seed)
eig_jacobi(aIn, maxSweeps)
svd(a)
null_vector(a)
pinv(a, tol)
rank(a, tol)
```

## mat4

Row-major flat 4x4 transforms for graphics: `translation`, `scaling`, `rot_x/y/z`, `rot_axis`, `trs`, `perspective(fovyDeg, ...)` (degrees!), `ortho`, `look_at`, `mul`, `transform_point` (w-divide), `transform_dir`, `transform_vec4`, `inverse`, `normal_matrix`.

```
identity()
get(m, r, c)
set(m, r, c, v)
mul(a, b)
transpose(m)
translation(tx, ty, tz)
scaling(sx, sy, sz)
rot_x(a)
rot_y(a)
rot_z(a)
rot_axis(axis, angle)
trs(t, eulerXYZ, s)
perspective(fovyDeg, aspect, near, far)
ortho(l, r, b, t, n, f)
look_at(eye, target, up)
transform_point(m, p)
transform_dir(m, d)
transform_vec4(m, p4)
to_nested(m)
from_nested(nm)
inverse(m)
determinant(m)
normal_matrix(m)
```

## quat

Quaternions `[w, x, y, z]`: `from_axis_angle`, `from_euler`, `qmul`, `rotate_vec`, `to_mat3`/`to_mat4`, `nlerp`, `slerp`, `from_to` (shortest arc), `conj`, `normalize`.

```
identity()
from_axis_angle(axis, angle)
from_euler(rx, ry, rz)
qmul(a, b)
conj(q)
qdot(a, b)
qnorm(q)
normalize(q)
rotate_vec(q, v)
to_mat4(q)
to_mat3(q)
nlerp(a, b, t)
slerp(a, b, t)
from_to(fromV, toV)
```

## tensor

N-dimensional arrays as `[shape, flatData]`: `tnew`, `reshape`, `tget`/`tset`, elementwise ops with shape checks, `tmatmul`, `ttranspose2d`, axis reductions, `tsoftmax_rows`, random fills, and bridges `to_matrix`/`from_matrix`.

```
tsize(shape)
tnew(shape, fill)
tzeros(shape)
tones(shape)
from_flat(shape, data)
shape(t)
data(t)
ndim(t)
numel(t)
strides(shp)
offset(shp, idx)
tget(t, idx)
tset(t, idx, v)
reshape(t, newShape)
trand(shape, seed, lo, hi)
trandn(shape, seed, scale)
tadd(a, b)
tsub(a, b)
tmul(a, b)
tscale(a, s)
tadd_scalar(a, s)
tapply(a, f)
tsum(a)
tmean(a)
tmin(a)
tmax(a)
targmax(a)
tsum_axis(t, axis)
tmean_axis(t, axis)
tslice0(t, i)
tstack(list)
tmatmul(a, b)
ttranspose2d(t)
tadd_rowvec(t, v)
tsoftmax_rows(t)
to_matrix(t)
from_matrix(m)
tprint(t, decimals)
```

## stats

Descriptive statistics: `mean`, `variance`/`stddev` (optional ddof), `median`, `quantile`, `mode_of`, `covariance`, Pearson `correlation`, `spearman`, `skewness`, `kurtosis`, `zscores`, `histogram` + `print_histogram`, `describe`, and Welch's t-test.

```
mean(xs)
variance(xs, ddof)
stddev(xs, ddof)
median(xs)
quantile(xs, q)
mode_of(xs)
covariance(xs, ys, ddof)
correlation(xs, ys)
spearman(xs, ys)
ranks(xs)
skewness(xs)
kurtosis(xs)
zscores(xs)
minmax_scale(xs)
histogram(xs, bins, lo, hi)
print_histogram(xs, bins, width)
describe(xs)
welch_t(a, b)
```

## dataset

Data plumbing: `read_text`/`write_text`, `load_csv`/`save_csv` (numeric features + optional string label column), `encode_labels`, `one_hot`, `train_test_split`, `kfold_indices`, standard/minmax scalers (fit on train, apply anywhere), and synthetic generators (`make_blobs`, `make_moons`, `make_regression`).

```
read_text(path)
write_text(path, contents)
parse_csv(text)
load_csv(path, hasHeader, targetCol)
save_csv(path, xrows, yvals, header)
shuffle_xy(xs, ys, seed)
train_test_split(xs, ys, testFrac, seed)
kfold_indices(n, k, seed)
fit_standardizer(xs)
apply_standardizer(xs, stdzr)
encode_labels(ys)
one_hot(ys, k)
make_blobs(nPerClass, dist, spread, seed)
make_moons(nPerClass, noise, seed)
make_regression(n, w, b, noise, seed)
make_xor(n, seed)
```

## mlcore

Shared ML machinery: distances (`euclidean`, `manhattan`, `cosine_similarity`, `minkowski`), activations with derivatives, `softmax`, losses (`mse`, `mae`, `bce`, `cce_row`), metrics (`accuracy`, `r2_score`, `confusion_matrix`, `precision_recall_f1`).

```
euclidean(a, b)
sq_euclidean(a, b)
manhattan(a, b)
chebyshev(a, b)
minkowski(a, b, p)
cosine_similarity(a, b)
cosine_distance(a, b)
sigmoid(x)
sigmoid_deriv(y)
relu(x)
relu_deriv(x)
leaky_relu(x)
leaky_relu_deriv(x)
tanh_act(x)
tanh_deriv(y)
softmax(xs)
mse(pred, target)
mae(pred, target)
bce(pred, target)
cce_row(pred, classIdx)
r2_score(pred, target)
rmse(pred, target)
accuracy(pred, target)
confusion_matrix(pred, target, k)
precision_recall_f1(cm)
predict_labels(scoreRows)
```

## ml_supervised

Classical supervised learning, each as `*_fit` returning a model value and `*_predict`: linear regression (QR least squares) + `ridge_fit` + gradient-descent variant, logistic regression, k-nearest neighbours, gaussian naive Bayes, perceptron, and a CART decision tree (`tree_fit(xs, ys, task, k, maxDepth, minSamples)` for classify or regress).

```
linreg_fit(xs, ys)
ridge_fit(xs, ys, lambda)
linreg_predict_one(model, x)
linreg_predict(model, xs)
linreg_gd_fit(xs, ys, lr, epochs)
logistic_fit(xs, ys, lr, epochs, l2)
logistic_prob_one(model, x)
logistic_predict(model, xs, threshold)
knn_model(xs, ys, k)
knn_predict_one(model, x)
knn_predict(model, xs)
gnb_fit(xs, ys, k)
gnb_predict_one(model, x)
gnb_predict(model, xs)
perceptron_fit(xs, ys, lr, epochs)
perceptron_predict(model, xs)
tree_fit(xs, ys, task, k, maxDepth, minSamples)
tree_predict_one(model, x)
tree_predict(model, xs)
tree_depth(model)
```

## ml_unsupervised

Unsupervised learning: `kmeans_fit` (k-means++ init), `pca_fit`/`pca_transform`/`pca_inverse` (components as columns), `dbscan`, agglomerative clustering with single/complete linkage, and `silhouette_score`.

```
kmeanspp_init(xs, k, seed)
kmeans_assign(xs, centers)
kmeans_fit(xs, k, maxIters, seed)
pca_fit(xs, nComponents)
pca_transform(model, xs)
pca_inverse(model, zs)
dbscan(xs, epsRadius, minPts)
agglomerative(xs, k)
```

## nn

Multi-layer perceptrons trained by explicit backprop. `build(sizes, activations, seed)`, `forward` returns a cache, `output`, `backward(net, cache, dLoss)` returns `[gradW, gradB, dInput]`, gradient utilities (`zero_grads`, `add_grads`, `scale_grads`), and optimizers `opt_sgd`/`opt_momentum`/`opt_adam` applied via `apply_grads`. Activations: relu, tanh, sigmoid, linear.

```
build(sizes, acts, seed)
act_apply(name, z)
act_deriv(name, z, a)
forward(net, x)
output(cache)
predict(net, x)
backward(net, cache, dOut)
zero_grads(net)
add_grads(accW, accB, gw, gb)
scale_grads(gw, gb, s)
opt_sgd(lr)
opt_momentum(net, lr, mu)
opt_adam(net, lr)
apply_grads(net, opt, gw, gb)
loss_grad(lossName, pred, target)
loss_value(lossName, pred, target)
train(net, opt, xs, ts, lossName, epochs, verboseEvery)
classify(net, x)
```

## cnn

Convolution building blocks on feature maps `[h, w, flatData]`: `conv2d` (valid cross-correlation) with `conv2d_backward` returning input/kernel/bias gradients (numerically verified), `relu_fmap` + backward, `maxpool2` + backward, `kernel_rand`, flatten helpers.

```
fmap(h, w, fill)
fmap_from(h, w, data)
fget(m, y, x)
kernel_rand(kh, kw, seed)
conv2d(input, kernel, bias)
conv2d_backward(input, kernel, dOut)
relu_fmap(m)
relu_fmap_backward(dOut, mask)
maxpool2(m)
maxpool2_backward(dOut, argm, ih, iw)
flatten_fmap(m)
unflatten_fmap(h, w, flat)
global_avg(m)
```

## image

Raster images `[w, h, channels, flatData]` with values in [0,1]. `new_image`, pixel access (`get_px`, `set_px`, clamped/bilinear sampling), `to_gray`, `resize` (bilinear), `upscale_nearest`, `crop`, flips, drawing (`fill_rect`, `draw_line`, `draw_circle`), ASCII preview `print_ascii`, and PPM/PGM save/load (`save` writes P2/P3, `load` reads P2/P3).

```
new_image(w, h, ch, fill)
width(img)
height(img)
channels(img)
idx_of(img, x, y, c)
get_px(img, x, y, c)
set_px(img, x, y, c, v)
get_px_clamped(img, x, y, c)
sample_bilinear(img, x, y, c)
clamp01(v)
to_gray(img)
to_rgb(img)
get_channel(img, c)
merge_rgb(r, g, b)
apply(img, f)
clamp_image(img)
scale_values(img, s, offset)
blend(a, b, t)
crop(img, x0, y0, cw, chh)
flip_h(img)
flip_v(img)
resize(img, nw, nh)
upscale_nearest(img, factor)
fill_rect(img, x0, y0, rw, rh, color)
draw_line(img, x0, y0, x1, y1, color)
draw_circle(img, cx, cy, radius, color)
save(img, path)
load(path)
print_ascii(img, chars)
```

## imgproc

Classic image processing on gray images: `convolve2d` with kernel builders (gaussian, box, sharpen, laplacian, emboss), `sobel`, full `canny` (blur, NMS, hysteresis), `threshold`/`otsu_threshold`/`equalize_hist`, binary morphology (`erode`, `dilate`, `opening`, `closing`), `connected_components`, `harris_corners`, and `match_template_ncc`.

```
convolve2d(img, kernel)
box_kernel(size)
gaussian_kernel(size, sigma)
sharpen_kernel()
laplacian_kernel()
emboss_kernel()
gaussian_blur(img, size, sigma)
sobel(img)
canny(img, lowT, highT)
threshold(img, t)
invert(img)
otsu_threshold(img)
equalize_hist(img)
erode(img, radius)
dilate(img, radius)
opening(img, radius)
closing(img, radius)
connected_components(img)
harris_corners(img, k, maxCorners)
match_template_ncc(img, tmpl)
```

## photo

Computational photography: `exposure`, `gamma_correct`, `reinhard_tonemap`, multi-exposure `merge_exposures` (hat-weighted HDR fusion), `gray_world_wb`/`scale_channels`, `unsharp_mask`, `bilateral_filter`, `median_filter`, `vignette`, `film_grain`, and Bayer sensor simulation (`bayer_mosaic`, `demosaic_bilinear`).

```
exposure(img, stops)
gamma_correct(img, gamma)
reinhard_tonemap(img, key, whitePoint)
merge_exposures(imgs, stops)
scale_channels(img, gains)
gray_world_wb(img)
unsharp_mask(img, amount, radius)
bilateral_filter(img, radius, sigmaS, sigmaR)
median_filter(img)
vignette(img, strength, falloff)
film_grain(img, amount, seed)
bayer_mosaic(img)
demosaic_bilinear(mos)
```

## texture

Procedural textures: `checker`, `stripes`, `gradient_tex`, hash-lattice `value_noise` / `fbm` / `turbulence` / `marble` / `wood`, `normal_from_height` (height map to tangent-space normal map), and wrapped bilinear `sample_uv` / `sample_uv_rgb`.

```
sample_uv(img, u, v, c)
sample_uv_rgb(img, u, v)
checker(w, h, cells, colorA, colorB)
stripes(w, h, count, horizontal, colorA, colorB)
gradient_tex(w, h, top2, bottom)
value_noise_at(x, y, seed)
fbm_at(x, y, octaves, seed)
value_noise(w, h, scale, octaves, seed)
turbulence(w, h, scale, octaves, seed)
marble(w, h, seed)
wood(w, h, rings, seed)
normal_from_height(heightImg, strength)
```

## mesh

Triangle meshes `[positions, normals, uvs, faces]` in one shared index space. Primitives: `cube` (24-vert, per-face UVs), `plane`, `uv_sphere`, `icosphere` (cached midpoint subdivision), `torus`, `cylinder`. Operations: `compute_normals` (area-weighted), `transform_mesh` (mat4 + normal matrix), `merge`, `bounds`, `surface_area`. I/O: `save_obj` / `load_obj` (v/vt/vn with corner-triple unification, fan triangulation).

```
new_mesh()
vertex_count(m)
face_count(m)
get_position(m, i)
get_normal(m, i)
get_uv(m, i)
get_face(m, f)
cube(s)
plane(sx, sz, nx, nz)
uv_sphere(radius, segments, rings)
icosphere(radius, subdivisions)
torus(bigR, tubeR, segments, sides)
cylinder(r, h, segments)
compute_normals(m)
transform_mesh(m, mat)
merge(a, b)
bounds(m)
surface_area(m)
save_obj(m, path)
load_obj(path)
parse_obj_corner(s)
```

## uvmap

UV unwrapping: `planar_project`, seam-aware `cylindrical_project` and `spherical_project` (wrap-seam triangles get duplicated +1 vertices), `box_project` (dominant-axis tri-planar), `unweld`, overlap-free per-face `atlas_pack_faces` (lightmap style), `uv_bounds`, `scale_uvs`, `uv_area_used`, and `checker_test_pattern` for visual inspection.

```
planar_project(m, axis)
cylindrical_project(m)
spherical_project(m)
fix_wrap_seam(m)
box_project(m)
unweld(m)
atlas_pack_faces(m, margin)
uv_bounds(m)
scale_uvs(m, su, sv)
uv_area_used(m)
checker_test_pattern(size, cells)
```

## render

A z-buffered software rasterizer: `make_target`, `draw_mesh` (model/view/projection mat4s, back-face culling, perspective-correct barycentric interpolation, Lambert + Blinn-Phong, optional texture), materials (`default_material`, `textured_material`), `default_light`, `orbit_camera` (fov in degrees), `depth_image`, `save_target`.

```
make_target(w, h, bg)
save_target(target, path)
default_material(color)
textured_material(tex)
default_light()
draw_mesh(target, m, model, view, proj, material, light)
orbit_camera(distance, yaw, pitch, fovDeg, aspect)
depth_image(target)
```

## nerf

A tiny neural radiance field trained end to end: positional encoding, MLP field (`make_field(L, hidden, seed, sigmaBias)` - negative bias starts transparent), emission-absorption `render_ray`, and `ray_grads` which pushes dL/dColor analytically through the compositing (suffix-sum) and the MLP (verified against numeric differentiation to 6 decimals). `train_rays` does one optimizer step over a ray batch; `camera_rays`, ground-truth `scene_*` helpers, `field_render_image`, `psnr`. Recipe: white background + `sigmaBias -2.0` + Adam lr 0.02.

```
posenc_dim(L)
posenc(p, L)
make_field(L, hidden, seed, sigmaBias)
softplus(x)
softplus_deriv(x)
query(field, p)
render_ray(field, ro, rd, near, far, nSamples, bg)
ray_grads(field, aux, dLdC)
camera_rays(eye, target, up, fovDeg, w, h)
scene_query(p)
scene_render_ray(ro, rd, near, far, nSamples, bg)
scene_render_image(eye, target, fovDeg, w, h, nSamples, bg)
field_render_image(field, eye, target, fovDeg, w, h, nSamples, bg)
train_rays(field, opt, origins, dirs, targets, near, far, nSamples, bg)
psnr(a, b)
```

## gsplat

3D gaussian splatting: splats `[pos, scale, quat, rgb, opacity]`, covariance `R S S^T R^T`, EWA screen-space projection `J W Sigma W^T J^T` (+0.3 low-pass), 3-sigma bounded rasterization with front-to-back alpha compositing, `make_camera`/`look_view`, initializers `from_points` and `from_mesh` (normal-aligned flattened splats per face), ASCII PLY `save_ply`/`load_ply`.

```
make_splat(pos, scale, rotQ, color, opacity)
covariance3d(scale, rotQ)
make_camera(eye, target, up, fovDeg, w, h)
look_view(eye, target, up)
render(splats, cam, bg)
from_points(points, colors, size, opacity)
from_mesh(m, opacity)
save_ply(splats, path)
load_ply(path)
```

## sfm

Two-view structure from motion: `intrinsics`, `project`, `normalize_point`, Hartley-normalized `eight_point` (fundamental, or essential with intrinsics; singular-value structure enforced after denormalization), `epipolar_error`, `recover_pose` (four candidates + cheirality vote), DLT `triangulate`, `reprojection_error`, rectified-stereo `disparity_map` (SAD block matching), and `save_ply_points`.

```
intrinsics(f, cx, cy)
project(k, r, t, x)
normalize_point(k, p)
hartley_normalize(pts)
eight_point(ptsA, ptsB, ka, kb)
epipolar_error(f, ptsA, ptsB)
recover_pose(e, ptsA, ptsB, k)
projection_matrix(k, r, t)
triangulate(ka, ra, ta, kb, rb, tb, xa, xb)
reprojection_error(k, r, t, points, pixels)
disparity_map(left, right, blockRadius, maxDisparity)
save_ply_points(points, colors, path)
```

## nlp

Text processing: `tokenize`/`sentences`, stopword removal, `ngrams`, a full Porter (1980) stemmer (verified on 50 reference pairs), vocabulary/bag-of-words/`tfidf_vectors`, `cosine_bow`, multinomial naive Bayes (`nb_fit`/`nb_predict`), lexicon `sentiment` with negation flipping, order-k Markov text generation, extractive `summarize`, and `levenshtein`.

```
tokenize(text)
sentences(text)
stopword_list()
remove_stopwords(tokens)
ngrams(tokens, n)
porter_stem(word)
stem_tokens(tokens)
build_vocab(docsTokens)
bow_vector(tokens, vocab)
tfidf_vectors(docsTokens)
cosine_bow(a, b)
nb_fit(docsTokens, labels)
nb_predict(model, tokens)
sentiment(text)
markov_fit(text, k)
markov_generate(model, nWords, seed)
summarize(text, nSentences)
levenshtein(a, b)
```

## embed

Count-based word embeddings - the factorization word2vec approximates: windowed `cooccurrence`, `ppmi` reweighting, symmetric eigendecomposition to dense vectors (`fit`), then `similarity`, `nearest`, and `analogy` (b - a + c).

```
cooccurrence(docsTokens, window)
ppmi(counts)
fit(docsTokens, window, dims)
vector(model, word)
cos(a, b)
similarity(model, wa, wb)
nearest(model, word, k)
nearest_vec(model, target, k, excludeWord)
analogy(model, a, b, c, k)
```

## physics

Simulation: generic `integrate_euler`/`integrate_semi_implicit`/`integrate_rk4` over user acceleration functions; particle systems with gravity, drag, and damped springs (`make_system`, `add_particle` with pinning, `add_spring`, `system_step`, `step_with_ground`); softened `n_body_step`; `kinetic_energy`/`momentum`; impulse `sphere_sphere` and `sphere_aabb` collisions; and `make_cloth` (structural + shear springs).

```
integrate_euler(accel, p, v, t, dt)
integrate_semi_implicit(accel, p, v, t, dt)
integrate_rk4(accel, p, v, t, dt)
make_system(gravity, drag)
add_particle(sys, pos, vel, mass)
add_spring(sys, i, j, stiffness, damping)
particle_count(sys)
get_particle(sys, i)
system_step(sys, dt)
step_with_ground(sys, dt, restitution, friction)
n_body_step(bodies, gConst, softening, dt)
kinetic_energy(velocities, masses)
momentum(velocities, masses)
sphere_sphere(a, b, restitution)
sphere_aabb(pos, vel, radius, boxMin, boxMax, restitution)
make_cloth(nx, ny, size, y0, pinTop, stiffness, damping)
```

## emeraldx

Package information: `version()`, `name()`, `modules()` inventory, and `info()` which prints the whole map.

```
version()
name()
modules()
info()
```
