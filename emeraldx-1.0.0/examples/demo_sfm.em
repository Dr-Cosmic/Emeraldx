# Structure from motion: two views -> relative pose -> 3D point cloud.
import sfm;
import matrix;
import vector;
import prng;
import strx;
import mathx;
int ang = 0;
int angErr = 0;
int cB = 0;
int cols = 0;
int i = 0;
int identA = 0;
int k = 0;
int pa = 0;
int pb = 0;
int pose = 0;
int pts3 = 0;
int r0 = 0;
int r1 = 0;
int r2 = 0;
int rB = 0;
int rEst = 0;
int rd = 0;
int rms = 0;
int sd = 0;
int tB = 0;
int tEst = 0;
int tr = 0;
int tri = 0;
int ua = 0;
int ub = 0;
int x = 0;
int z = 0;
int zeroT = 0;
# emx-scope-safe

print("== emeraldx: structure from motion ==");
k = sfm.intrinsics(140.0, 80.0, 60.0);
ang = 0.28;
rB = [[cos(ang), 0.0, sin(ang)], [0.0, 1.0, 0.0], [0.0 - sin(ang), 0.0, cos(ang)]];
cB = [1.2, 0.2, -0.25];
tB = vector.vscale(matrix.matvec(rB, cB), -1.0);
identA = matrix.identity(3);
zeroT = [0.0, 0.0, 0.0];

sd = prng.seed_from(19);
pts3 = []; pa = []; pb = [];
while (len(pts3) < 16) {
    r0 = prng.uniform(sd, -1.8, 1.8); sd = r0[1];
    r1 = prng.uniform(sd, -1.3, 1.3); sd = r1[1];
    r2 = prng.uniform(sd, 3.5, 8.0); sd = r2[1];
    x = [r0[0], r1[0], r2[0]];
    ua = sfm.project(k, identA, zeroT, x);
    ub = sfm.project(k, rB, tB, x);
    if (ua == nil || ub == nil) { continue; }
    pts3[len(pts3)] = x;
    pa[len(pa)] = ua;
    pb[len(pb)] = ub;
}
print("correspondences   ", len(pa));
e = sfm.eight_point(pa, pb, k, k);
pose = sfm.recover_pose(e, pa, pb, k);
rEst = pose[0]; tEst = pose[1];
print("cheirality        ", pose[2], "/", len(pa), " points in front of both cameras");

rd = matrix.matmul(rEst, matrix.transpose(rB));
tr = rd[0][0] + rd[1][1] + rd[2][2];
angErr = acos(mathx.clampf((tr - 1.0) / 2.0, -1.0, 1.0)) * 180.0 / pi;
print("rotation error    ", strx.fmt_f(angErr, 4), " deg");
print("translation dot   ", strx.fmt_f(abs(vector.vdot(vector.vnormalize(tEst), vector.vnormalize(tB))), 4));

tri = [];
cols = [];
for (i = 0; i < len(pa); ++i) {
    tri[i] = sfm.triangulate(k, identA, zeroT, k, rEst, tEst, pa[i], pb[i]);
    z = tri[i][2];
    cols[i] = [mathx.clampf(1.2 - z * 0.12, 0.0, 1.0), 0.4, mathx.clampf(z * 0.12, 0.0, 1.0)];
}
rms = sfm.reprojection_error(k, identA, zeroT, tri, pa);
print("reprojection rms  ", strx.fmt_f(rms, 5), " px");
sfm.save_ply_points(tri, cols, "examples/out/sfm_cloud.ply");
print("wrote examples/out/sfm_cloud.ply (", len(tri), " points)");
