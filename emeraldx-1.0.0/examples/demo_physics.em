# Physics: integrator accuracy, orbits, bouncing, and a cloth that becomes
# a mesh you can open in any 3D tool.
import physics;
import vector;
import mesh;
import strx;
import mathx;
int apex = 0;
int bodies = 0;
int bot = 0;
int cloth = 0;
int dt = 0;
int i = 0;
int m = 0;
int nx = 0;
int ny = 0;
int p = 0;
int p0 = 0;
int p1 = 0;
int pe = 0;
int positions = 0;
int pp = 0;
int r = 0;
int r2 = 0;
int steps = 0;
int sys = 0;
int top2 = 0;
int v = 0;
int ve = 0;
# emx-scope-safe

print("== emeraldx: physics ==");
# integrator accuracy on simple harmonic motion over one period
fn shm(p, v, t) { return [0.0 - p[0], 0.0, 0.0]; }
p = [1.0, 0.0, 0.0]; v = [0.0, 0.0, 0.0];
steps = 126;
dt = mathx.TAU / steps;
pe = p; ve = v;
for (i = 0; i < steps; ++i) {
    r = physics.integrate_rk4(shm, p, v, i * dt, dt);
    p = r[0]; v = r[1];
    r2 = physics.integrate_euler(shm, pe, ve, i * dt, dt);
    pe = r2[0]; ve = r2[1];
}
print("SHM after 1 period   rk4 x=", strx.fmt_f(p[0], 6), "  euler x=", strx.fmt_f(pe[0], 4), "  (exact 1.0)");

# two-body orbit conserves momentum
bodies = [[-1.0,0.0,0.0, 1.0,0.0,0.0], [0.0,-0.35,0.0, 0.0,0.35,0.0], [1.0, 1.0]];
p0 = physics.momentum(bodies[1], bodies[2]);
for (i = 0; i < 400; ++i) { bodies = physics.n_body_step(bodies, 0.5, 0.01, 0.01); }
p1 = physics.momentum(bodies[1], bodies[2]);
print("orbit momentum drift ", strx.fmt_f(vector.vdist(p0, p1), 9));

# bouncing ball apex ratio ~ restitution^2
sys = physics.make_system(nil, 0.0);
r = physics.add_particle(sys, [0.0, 2.0, 0.0], [0.0, 0.0, 0.0], 1.0);
sys = r[0];
apex = 0.0;
for (i = 0; i < 300; ++i) {
    sys = physics.step_with_ground(sys, 0.01, 0.6, 1.0);
    pp = physics.get_particle(sys, 0);
    if (i > 80 && pp[1] > apex) { apex = pp[1]; }
}
print("bounce apex          ", strx.fmt_f(apex, 3), " of 2.0  (0.6^2 = 0.36 ratio)");

# cloth: simulate, then export the deformed grid as an OBJ mesh
nx = 6; ny = 6;
cloth = physics.make_cloth(nx, ny, 1.2, 1.2, true, 90.0, 0.8);
for (i = 0; i < 500; ++i) { cloth = physics.system_step(cloth, 0.006); }
positions = cloth[0];
m = mesh.plane(1.2, 1.2, nx, ny);
m = [positions, [], m[2], m[3]];
m = mesh.compute_normals(m);
mesh.save_obj(m, "examples/out/cloth.obj");
top2 = physics.get_particle(cloth, 3);
bot = physics.get_particle(cloth, (ny) * (nx + 1) + 3);
print("cloth top/bottom y   ", strx.fmt_f(top2[1], 2), " / ", strx.fmt_f(bot[1], 2));
print("wrote examples/out/cloth.obj (", mesh.vertex_count(m), " verts)");
