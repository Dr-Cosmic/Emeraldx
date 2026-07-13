# emeraldx

Author: Jean-Jacques François Reibel

An AI / machine learning / computer vision / 3D graphics library for the
[Emerald](https://github.com/Dr-Cosmic/Emerald) scripting language - 31
modules of numerics, learning, imaging, and geometry written in pure Emerald,
with no dependencies beyond the interpreter itself.

Highlights:

- **Neural networks** trained by explicit backprop (SGD / momentum / Adam),
  plus convolution and pooling layers with numerically verified gradients.
- **A tiny NeRF**: a neural radiance field whose training gradients flow
  *analytically* through the emission-absorption volume renderer (checked
  against numeric differentiation to six decimals) - it learns a 3D scene
  from posed images inside an interpreted scripting language.
- **3D gaussian splatting**: quaternion-scale covariances, EWA screen-space
  projection, depth-sorted alpha compositing, PLY interchange.
- **Photogrammetry**: the normalized eight-point algorithm, essential-matrix
  pose recovery with cheirality testing, DLT triangulation, and stereo block
  matching - machine-precision on synthetic data.
- **UV unwrapping & texturing**: planar / cylindrical / spherical / box
  projections with wrap-seam repair, an overlap-free lightmap atlas packer,
  procedural textures (fbm, marble, wood, normal maps), and a z-buffered
  perspective-correct software rasterizer that renders textured meshes.
- **Classical ML**: linear/logistic regression, kNN, naive Bayes, decision
  trees, k-means++, PCA, DBSCAN - with dataset utilities and metrics.
- **Computational photography**: HDR exposure fusion, Reinhard tonemapping,
  white balance, bilateral/median denoising, Bayer demosaicing.
- **NLP**: a full Porter stemmer (50/50 on the classic reference pairs),
  TF-IDF, naive Bayes text classification, Markov generation, extractive
  summarization, and PPMI word embeddings with analogy queries.
- **Physics**: RK4/semi-implicit integrators, spring-mass systems, N-body
  gravity, impulse collisions, and cloth simulation that exports to OBJ.
- Foundations: vectors, dense matrices (QR, eigen, SVD, pseudoinverse),
  quaternions, 4x4 transforms, tensors, FFT, statistics, PRNG, strings,
  dictionaries.

## Requirements

The Emerald interpreter, built from https://github.com/Dr-Cosmic/Emerald.
Nothing else.

## Quick start

Module resolution in Emerald is relative to the **current working
directory**, so always run from the package root:

```sh
cd emeraldx
emerald selftest.em            # 32 checks, one per module - takes seconds
emerald examples/demo_render.em
```

Then, in your own script (also run from this directory):

```
import mesh;
import uvmap;
import render;
import texture;
import mat4;

cube = mesh.cube(1.5);
cam = render.orbit_camera(4.0, 0.7, 0.5, 45.0, 4.0 / 3.0);
target = render.make_target(80, 60, nil);
target = render.draw_mesh(target, cube, mat4.rot_y(0.4), cam[0], cam[1],
                          render.textured_material(uvmap.checker_test_pattern(32, 4)),
                          render.default_light());
render.save_target(target, "cube.ppm");
```

## Examples

Each demo is self-contained, prints what it verifies, and writes any images,
OBJ, or PLY artifacts to `examples/out/`:

| script | shows |
|---|---|
| `examples/demo_linalg.em` | solve, least squares, eigen, SVD, quaternion+mat4 pipeline |
| `examples/demo_ml.em` | regression R2, three classifiers at 1.0 accuracy, k-means, PCA |
| `examples/demo_nn_xor.em` | an MLP learning XOR with Adam |
| `examples/demo_cv.em` | Otsu, connected components, Canny, Harris corners |
| `examples/demo_photo.em` | HDR merge, tonemap, white balance, denoising, demosaic |
| `examples/demo_mesh_uv.em` | icosphere, spherical unwrap with seam repair, atlas, OBJ round trip |
| `examples/demo_render.em` | textured cube + torus, z-buffer, depth image |
| `examples/demo_nerf.em` | trains a NeRF on 3 views, renders a novel view (~1 min) |
| `examples/demo_gsplat.em` | gaussian splat ring, mesh->splats, PLY round trip |
| `examples/demo_sfm.em` | two-view pose recovery + triangulated point cloud |
| `examples/demo_nlp.em` | stemming, classification, sentiment, Markov, summary, embeddings |
| `examples/demo_physics.em` | integrator accuracy, orbits, bouncing, cloth -> OBJ |

## The 31 modules

List them from Emerald itself (`import emeraldx; emeraldx.info();` in any
script) - or see the same map with full function signatures in `docs/API.md`.

core: `mathx` `prng` `arrayx` `strx` `dictx` `complexx`
linear algebra: `vector` `matrix` `mat4` `quat` `tensor`
data & ML: `stats` `dataset` `mlcore` `ml_supervised` `ml_unsupervised` `nn` `cnn`
imaging: `image` `imgproc` `photo` `texture`
3D: `mesh` `uvmap` `render` `nerf` `gsplat` `sfm`
language & simulation: `nlp` `embed` `physics` - plus the `emeraldx` info module.

## Emerald-specific notes

These matter when reading or extending the library; the full story is in
`docs/DESIGN.md`.

- **Values everywhere.** Arrays (and everything built from them - matrices,
  images, meshes, networks) copy on assignment and on function call. All
  APIs therefore *return* updated state: `d = dictx.dset(d, k, v);`,
  `sys = physics.system_step(sys, dt);`, `net/opt` come back from
  `nn.apply_grads`.
- **Scope safety.** Assignment in Emerald writes to an existing outer
  variable of the same name unless the name was declared locally. Every
  function in this library therefore opens with generated
  `int name = 0;` declarations (the `# emx-scope-safe` marker) so that
  library internals can never collide with *your* globals - verified by a
  stress test that runs the ML stack under 29 deliberately colliding global
  names.
- **Files.** `File f = pathExpression;` binds a path; `f.read()` returns the
  whole file, `f.write(s)` replaces it, `f.append(s)` appends. There is no
  close and no mode string.
- **Optional arguments.** Trailing arguments may be omitted; they arrive as
  `nil` and functions substitute documented defaults.
- **Performance.** The interpreter executes roughly a few million simple
  operations per second. Everything here is sized for that: demos use small
  images (48-80 px), small meshes, and short training runs. The algorithms
  are real; scale dimensions with care.

## Layout

```
emeraldx/
  *.em              the 31 modules + selftest.em
  examples/         12 runnable demos (write artifacts to examples/out/)
  docs/API.md       per-module function reference
  docs/DESIGN.md    data formats, conventions, and Emerald findings
  LICENSE           MIT
```

## License

MIT - see `LICENSE`.
