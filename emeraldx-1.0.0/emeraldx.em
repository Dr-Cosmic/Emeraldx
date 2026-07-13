# emx-scope-safe
# =============================================================================
# emeraldx.em - library information module
#
#   import emeraldx;
#   emeraldx.info();          print name, version and module inventory
#   v = emeraldx.version();   "1.0.0"
#   m = emeraldx.modules();   array of [name, description]
# =============================================================================

fn version() { return "1.0.0"; }

fn name() { return "emeraldx"; }

fn modules() {
    return [
        ["mathx",           "scalar math: constants, clamp, atan2, erf, gamma, roots"],
        ["prng",            "deterministic random: uniform, normal, shuffle, Rng class"],
        ["arrayx",          "array utilities: map-style helpers, sort, argsort, stats"],
        ["strx",            "strings: split, join, trim, parse, format"],
        ["dictx",           "string-keyed dictionary on sorted parallel arrays"],
        ["complexx",        "complex numbers and radix-2 FFT"],
        ["vector",          "n-dim and 3D vectors: dot, cross, norms, lerp"],
        ["matrix",          "dense matrices: LU, QR, least squares, eig, SVD, pinv"],
        ["mat4",            "4x4 transforms: TRS, perspective, look_at, inverse"],
        ["quat",            "quaternions: axis-angle, slerp, rotation matrices"],
        ["tensor",          "n-dim arrays: reshape, transpose, broadcast, matmul"],
        ["stats",           "descriptive stats, covariance, histograms, t-test"],
        ["dataset",         "CSV I/O, label encoding, splits, k-fold, scalers"],
        ["mlcore",          "losses, metrics, activations, confusion matrix"],
        ["ml_supervised",   "linear/logistic regression, kNN, naive Bayes, trees"],
        ["ml_unsupervised", "k-means++, PCA, DBSCAN, agglomerative clustering"],
        ["nn",              "MLPs: backprop, SGD/momentum/Adam, gradient utils"],
        ["cnn",             "conv2d + maxpool layers with verified backward"],
        ["image",           "raster images, PPM/PGM I/O, resize, draw, ASCII"],
        ["imgproc",         "convolution, Canny, Otsu, morphology, Harris, NCC"],
        ["photo",           "HDR merge, tonemap, white balance, bilateral, demosaic"],
        ["texture",         "procedural textures: noise, fbm, marble, normal maps"],
        ["mesh",            "triangle meshes: primitives, normals, OBJ I/O"],
        ["uvmap",           "UV unwrapping: planar/cyl/sphere/box, seams, atlas"],
        ["render",          "software rasterizer: z-buffer, Blinn-Phong, textures"],
        ["nerf",            "tiny neural radiance field with analytic gradients"],
        ["gsplat",          "3D gaussian splatting renderer + PLY I/O"],
        ["sfm",             "structure from motion: 8-point, pose, triangulation"],
        ["nlp",             "tokenize, Porter stemmer, TF-IDF, NB, Markov, summary"],
        ["embed",           "PPMI + eigen word embeddings, analogies"],
        ["physics",         "integrators, particles, springs, n-body, cloth"]
    ];
}

fn info() {
    int i = 0; int m = 0;
    print(name(), " ", version(), " - AI / ML / CV / 3D library for Emerald");
    m = modules();
    for (i = 0; i < len(m); ++i) {
        e = m[i];
        print("  ", e[0], " - ", e[1]);
    }
    print(len(m), " modules. Run from the package root; see README.md.");
    return true;
}
