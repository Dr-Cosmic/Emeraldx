# Linear algebra tour: solve, least squares, eigen, SVD, quaternions, mat4.
import matrix;
import vector;
import mat4;
import quat;
import strx;
int a = 0;
int b = 0;
int d = 0;
int eg = 0;
int err = 0;
int i = 0;
int j = 0;
int m = 0;
int mfull = 0;
int mrot = 0;
int p = 0;
int q = 0;
int r = 0;
int rec = 0;
int s = 0;
int sg = 0;
int sv = 0;
int u = 0;
int v = 0;
int w = 0;
int x = 0;
int xs = 0;
int ys = 0;
# emx-scope-safe

print("== emeraldx: linear algebra ==");
a = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
b = [1.0, 2.0, 3.0];
x = matrix.solve(a, b);
print("solve Ax=b        x = ", strx.fmt_arr(x, 4, ", "));
r = vector.vsub(matrix.matvec(a, x), b);
print("residual |Ax-b|   ", strx.fmt_f(vector.vnorm(r), 10));

# overdetermined least squares: fit y = 2x + 1 with noise-free samples
xs = [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]];
ys = [1.0, 3.0, 5.0, 7.0];
w = matrix.lstsq(xs, ys);
print("lstsq slope/bias  ", strx.fmt_arr(w, 4, ", "), "  (want 2, 1)");

# eigen of a symmetric matrix
s = [[2.0, 1.0], [1.0, 2.0]];
eg = matrix.eig_jacobi(s, nil);
print("eigenvalues       ", strx.fmt_arr(eg[0], 4, ", "), "  (want 3, 1)");

# SVD reconstruction error
m = [[3.0, 1.0], [2.0, 2.0], [0.0, 4.0]];
sv = matrix.svd(m);
u = sv[0]; sg = sv[1]; v = sv[2];
rec = matrix.matmul(u, matrix.matmul(matrix.diag(sg), matrix.transpose(v)));
err = 0.0;
for (i = 0; i < 3; ++i) {
    for (j = 0; j < 2; ++j) {
        d = rec[i][j] - m[i][j];
        if (d < 0.0) { d = 0.0 - d; }
        if (d > err) { err = d; }
    }
}
print("SVD recon err     ", strx.fmt_f(err, 10));

# quaternion + mat4 pipeline: rotate then translate a point
q = quat.from_axis_angle([0.0, 1.0, 0.0], pi / 2.0);
mrot = quat.to_mat4(q);
mfull = mat4.mul(mat4.translation(10.0, 0.0, 0.0), mrot);
p = mat4.transform_point(mfull, [1.0, 0.0, 0.0]);
print("quat+mat4 point   ", strx.fmt_arr(p, 3, ", "), "  (want 10, 0, -1)");
