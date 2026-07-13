# emeraldx design notes

How the library is put together, the data formats every module agrees on,
and the Emerald language findings that shaped the code. Read this before
extending the library.

## 1. Data formats

Everything is built from Emerald's three workhorses - numbers, strings, and
arrays - because arrays have **value semantics**: assignment and argument
passing copy. There are no reference types to alias, so every API returns
its updated state.

| thing | layout |
|---|---|
| vector | flat array `[x, y, z, ...]` |
| matrix | nested rows `[[a,b],[c,d]]` (module `matrix`) |
| mat4 | flat row-major 16 floats (module `mat4`) |
| quaternion | `[w, x, y, z]` |
| tensor | `[shape, flatData]`, row-major strides |
| dict | `[sortedKeys, values]`, binary search (module `dictx`) |
| complex | `[re, im]`; FFT takes separate re/im arrays |
| image | `[w, h, channels, flatData]`, row-major, channel-interleaved, values 0..1 |
| feature map (cnn) | `[h, w, flatData]` |
| mesh | `[positions, normals, uvs, faces]` - parallel flat arrays over ONE vertex index space; `faces` is flat triangle indices, CCW |
| splat | `[pos3, scale3, quat4, rgb3, opacity]` |
| neural net | `[sizes, activations, weights, biases]`; `weights[l]` is rows-of-neurons |
| optimizer | opaque value from `nn.opt_*`, returned updated by `nn.apply_grads` |
| particle system | `[positions, velocities, invMasses, springs, gravity, drag]` |
| PRNG state | integer; every draw returns `[value, newState]` |

Conventions:

- Angles are radians everywhere **except** `mat4.perspective(fovyDeg, ...)`
  and `render.orbit_camera(..., fovDeg, ...)`, which take degrees (as their
  parameter names say).
- Cameras look down **-z** (OpenGL style). `sfm` follows the computer-vision
  convention instead: `x = K [R | t] X` with the camera looking down **+z**.
- Images put (0,0) at the top-left; UV (0,0) is also top-left and
  `texture.sample_uv` wraps.
- Meshes triangulate on import; OBJ corner triples `v/vt/vn` are unified
  into the single shared index space (vertex order may change; geometry,
  areas, and bounds are preserved exactly).

## 2. Scope safety (important!)

Emerald's assignment rule: `name = value;` inside a function assigns to an
*existing* variable named `name` in any enclosing scope - including your
script's globals - unless the function declared its own. A typed declaration
like `int name = 0;` unconditionally creates a fresh local (which may later
hold any type: Emerald does not re-coerce).

A library that used bare assignment internally could silently overwrite the
caller's globals (`i`, `x`, `out`, `img`, ...). Therefore **every function
in emeraldx begins with generated `int name = 0;` declarations** for each
name it assigns - files carry a `# emx-scope-safe` marker once processed.
The transformer that maintains this lives outside the package
(`scopesafe.py` in the development tree); if you add functions by hand,
declare your locals the same way.

Class fields are the one exception: fields are created by plain assignment
in methods, so a field named like a user global would collide. The single
stateful class in the library (`prng.Rng`) names its field `emxState` to
make collisions implausible.

This is stress-tested: the self-check and the ML test suite run under a
namespace of 29 deliberately colliding globals and verify none change.

## 3. Emerald findings the code relies on

Verified against the interpreter (Dr-Cosmic/Emerald) during development:

- **Value semantics** for arrays in assignment *and* calls; `arrayx.pop(a)`
  therefore returns `[rest, popped]`.
- **Missing call arguments become `nil`**, so optional trailing parameters
  work: `if (x == nil) { x = default; }`. Extra arguments are ignored.
- **Function values** exist: `mathx.bisect(f, ...)`,
  `physics.integrate_rk4(accel, ...)`, `arrayx.map(a, f)` all take functions.
- **String interpolation** `"{name}"` substitutes in-scope variables inside
  string literals (used by the OBJ/PLY writers).
- **`File f = pathExpr;`** is the file API: `read()` whole file, `write(s)`
  replace, `append(s)` append. `File f(path, mode)` is NOT a constructor
  call - it would create a file literally named `f`. Writers build one big
  string (or append row-wise) accordingly.
- **Blocks are child scopes**: a declaration inside `{ }` vanishes at the
  brace; the generated declarations therefore sit at function top level.
- `for (int i = 0; ...)` parses; loop variables of `for x in arr` are
  already local; redeclaration in the same scope is allowed.
- Integer division is `//`; `top()` is ceiling; `ln`/`log` are natural/base-10.
- The interpreter buffers stdout: a run killed by a timeout shows nothing,
  so absence of output never means absence of progress.

## 4. Numerical choices

- `matrix.svd` works via eigen-decomposition of `A^T A` (Jacobi sweeps).
  Two hard-won details: the rank cutoff is **relative** to the largest
  singular value (absolute epsilons misclassify roundoff-sized values on
  well-scaled matrices), and rank-deficient columns of `U` are **completed
  to an orthonormal basis** by Gram-Schmidt - consumers like essential-matrix
  decomposition read the left null space from `U`'s last column and build
  rotations as `U W V^T`, which must be orthogonal. Note `svd` returns
  `[U, s, V]` (V, not V^T).
- `sfm.eight_point` Hartley-normalizes for conditioning, solves with
  `matrix.null_vector`, **denormalizes first**, and only then enforces the
  singular-value structure (rank-2 for F; equal pair for E) - that
  structure does not survive the normalization transforms.
- The NeRF backward pass differentiates the compositing exactly using a
  suffix sum over downstream contributions:
  `dL/dalpha_i = (dLdC . c_i) T_i - S_i / (1 - alpha_i)`, then
  `dL/dsigma_i = dL/dalpha_i * dt * (1 - alpha_i)`; rgb heads chain through
  sigmoid, density through softplus. Verified against numeric gradients.
  Training recipe that converges quickly: white background + transparent
  initialization (`sigmaBias = -2`) + Adam at lr 0.02.
- `gsplat` follows the 3DGS math: `Sigma = R S S^T R^T`, screen covariance
  `J W Sigma W^T J^T` plus the paper's 0.3 low-pass, splat extents bounded
  by 3 sigma of the largest eigenvalue, strict front-to-back compositing
  with early termination at transmittance 0.004.
- `imgproc.otsu_threshold` returns `(bestBin + 1) / 255` - the upper edge of
  the background bin - so thresholding with `>=` separates exactly.
- Where reference results exist, tests pin them: Porter stemmer 50/50
  classic pairs, XOR to loss < 1e-4, conv2d/maxpool gradients to 5 decimals,
  RK4 on simple harmonic motion to 6 decimals over a full period, k-means
  centers, PCA directions, HDR reconstruction error, OBJ/PLY round trips.

## 5. Performance envelope

The interpreter runs on the order of a few million simple operations per
second. Consequences baked into the defaults:

- demo images are 48-80 px; the NeRF demo trains 8x8 views with 8 samples
  per ray and a 16-wide MLP (~2.5 s/epoch);
- `matrix` routines are fine into the tens of columns, not thousands;
- prefer building one big string over many concatenations in inner loops
  (string concat is O(n)); file writers append per row.

Everything scales up mechanically if you have the patience - the algorithms
are the real ones.
