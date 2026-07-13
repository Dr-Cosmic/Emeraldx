# =============================================================================
# physics.em - simulation toolkit
#
#   integrate_euler / integrate_semi_implicit / integrate_rk4
#       generic ODE steppers; accel(pos, vel, t) is a user function
#   Particle systems:
#       make_system, add_particle, add_spring
#       system_step (gravity + drag + springs, semi-implicit)
#   n_body_step                      pairwise gravitation
#   Collisions:
#       sphere_sphere, sphere_aabb resolution helpers
#       step_with_ground             bounce off y = 0 plane
#   make_cloth                       grid of particles + structural/shear
#                                    springs, ready for system_step
#   kinetic_energy, momentum
#
# state layout: positions/velocities are flat [x,y,z, ...] arrays; a system is
#   [positions, velocities, invMasses, springs, gravity, drag]
# a spring is [i, j, restLength, stiffness, damping]
# =============================================================================

import mathx;
import arrayx;
import vector;
# emx-scope-safe

# ---------------------------------------------------------------------------
# Generic single-body integrators. accel(p, v, t) -> acceleration vector.
# state = [p, v]; returns the new [p, v].
# ---------------------------------------------------------------------------

fn integrate_euler(accel, p, v, t, dt) {
    int a = 0; int np = 0; int nv = 0;
    a = accel(p, v, t);
    np = vector.vadd(p, vector.vscale(v, dt));
    nv = vector.vadd(v, vector.vscale(a, dt));
    return [np, nv];
}

fn integrate_semi_implicit(accel, p, v, t, dt) {
    int a = 0; int np = 0; int nv = 0;
    a = accel(p, v, t);
    nv = vector.vadd(v, vector.vscale(a, dt));
    np = vector.vadd(p, vector.vscale(nv, dt));
    return [np, nv];
}

# Classic RK4 on the coupled (p, v) system.
fn integrate_rk4(accel, p, v, t, dt) {
    int a1 = 0; int a2 = 0; int a3 = 0; int a4 = 0; int i = 0; int k1p = 0; int k1v = 0; int k2p = 0; int k2v = 0; int k3p = 0; int k3v = 0; int k4p = 0; int k4v = 0; int np = 0; int nv = 0; int p2 = 0; int p3 = 0; int p4 = 0; int v2 = 0; int v3 = 0; int v4 = 0;
    a1 = accel(p, v, t);
    k1p = v;
    k1v = a1;

    p2 = vector.vadd(p, vector.vscale(k1p, dt * 0.5));
    v2 = vector.vadd(v, vector.vscale(k1v, dt * 0.5));
    a2 = accel(p2, v2, t + dt * 0.5);
    k2p = v2;
    k2v = a2;

    p3 = vector.vadd(p, vector.vscale(k2p, dt * 0.5));
    v3 = vector.vadd(v, vector.vscale(k2v, dt * 0.5));
    a3 = accel(p3, v3, t + dt * 0.5);
    k3p = v3;
    k3v = a3;

    p4 = vector.vadd(p, vector.vscale(k3p, dt));
    v4 = vector.vadd(v, vector.vscale(k3v, dt));
    a4 = accel(p4, v4, t + dt);
    k4p = v4;
    k4v = a4;

    np = [];
    nv = [];
    for (i = 0; i < len(p); ++i) {
        np[i] = p[i] + dt / 6.0 * (k1p[i] + 2.0 * k2p[i] + 2.0 * k3p[i] + k4p[i]);
        nv[i] = v[i] + dt / 6.0 * (k1v[i] + 2.0 * k2v[i] + 2.0 * k3v[i] + k4v[i]);
    }
    return [np, nv];
}

# ---------------------------------------------------------------------------
# Particle system
# ---------------------------------------------------------------------------

fn make_system(gravity, drag) {
    if (gravity == nil) { gravity = [0.0, -9.81, 0.0]; }
    if (drag == nil) { drag = 0.02; }
    return [[], [], [], [], gravity, drag];
}

# mass <= 0 pins the particle (infinite mass). Returns [system, index].
fn add_particle(sys, pos, vel, mass) {
    int idx = 0; int invM = 0; int positions = 0; int velocities = 0;
    positions = sys[0];
    velocities = sys[1];
    invM = sys[2];
    idx = len(positions) // 3;
    positions = positions + [pos[0], pos[1], pos[2]];
    velocities = velocities + [vel[0], vel[1], vel[2]];
    if (mass <= 0.0) { invM[idx] = 0.0; }
    else { invM[idx] = 1.0 / mass; }
    return [[positions, velocities, invM, sys[3], sys[4], sys[5]], idx];
}

fn add_spring(sys, i, j, stiffness, damping) {
    int pi2 = 0; int pj = 0; int positions = 0; int rest = 0; int springs = 0;
    positions = sys[0];
    pi2 = [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];
    pj = [positions[j * 3], positions[j * 3 + 1], positions[j * 3 + 2]];
    rest = vector.vdist(pi2, pj);
    springs = sys[3];
    springs[len(springs)] = [i, j, rest, stiffness, damping];
    return [sys[0], sys[1], sys[2], springs, sys[4], sys[5]];
}

fn particle_count(sys) { return len(sys[0]) // 3; }

fn get_particle(sys, i) {
    int positions = 0;
    positions = sys[0];
    return [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];
}

# One semi-implicit step of gravity + linear drag + springs.
fn system_step(sys, dt) {
    int dist = 0; int drag = 0; int dx = 0; int dy = 0; int dz = 0; int fmag = 0; int forces = 0; int g = 0; int i = 0; int im = 0; int invM = 0; int j = 0; int kd = 0; int ks = 0; int m = 0; int n = 0; int positions = 0; int relv = 0; int rest = 0; int s = 0; int sp = 0; int springs = 0; int ux = 0; int uy = 0; int uz = 0; int velocities = 0;
    positions = sys[0];
    velocities = sys[1];
    invM = sys[2];
    springs = sys[3];
    g = sys[4];
    drag = sys[5];
    n = len(positions) // 3;
    forces = arrayx.zerosf(n * 3);
    # gravity + drag
    for (i = 0; i < n; ++i) {
        if (invM[i] == 0.0) { continue; }
        m = 1.0 / invM[i];
        forces[i * 3] = g[0] * m - drag * velocities[i * 3];
        forces[i * 3 + 1] = g[1] * m - drag * velocities[i * 3 + 1];
        forces[i * 3 + 2] = g[2] * m - drag * velocities[i * 3 + 2];
    }
    # springs (Hooke + relative-velocity damping along the axis)
    for (s = 0; s < len(springs); ++s) {
        sp = springs[s];
        i = sp[0];
        j = sp[1];
        rest = sp[2];
        ks = sp[3];
        kd = sp[4];
        dx = positions[j * 3] - positions[i * 3];
        dy = positions[j * 3 + 1] - positions[i * 3 + 1];
        dz = positions[j * 3 + 2] - positions[i * 3 + 2];
        dist = sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < 1.0e-9) { continue; }
        ux = dx / dist;
        uy = dy / dist;
        uz = dz / dist;
        relv = (velocities[j * 3] - velocities[i * 3]) * ux +
               (velocities[j * 3 + 1] - velocities[i * 3 + 1]) * uy +
               (velocities[j * 3 + 2] - velocities[i * 3 + 2]) * uz;
        fmag = ks * (dist - rest) + kd * relv;
        forces[i * 3] = forces[i * 3] + fmag * ux;
        forces[i * 3 + 1] = forces[i * 3 + 1] + fmag * uy;
        forces[i * 3 + 2] = forces[i * 3 + 2] + fmag * uz;
        forces[j * 3] = forces[j * 3] - fmag * ux;
        forces[j * 3 + 1] = forces[j * 3 + 1] - fmag * uy;
        forces[j * 3 + 2] = forces[j * 3 + 2] - fmag * uz;
    }
    # semi-implicit integrate
    for (i = 0; i < n; ++i) {
        im = invM[i];
        if (im == 0.0) { continue; }
        velocities[i * 3] = velocities[i * 3] + forces[i * 3] * im * dt;
        velocities[i * 3 + 1] = velocities[i * 3 + 1] + forces[i * 3 + 1] * im * dt;
        velocities[i * 3 + 2] = velocities[i * 3 + 2] + forces[i * 3 + 2] * im * dt;
        positions[i * 3] = positions[i * 3] + velocities[i * 3] * dt;
        positions[i * 3 + 1] = positions[i * 3 + 1] + velocities[i * 3 + 1] * dt;
        positions[i * 3 + 2] = positions[i * 3 + 2] + velocities[i * 3 + 2] * dt;
    }
    return [positions, velocities, invM, springs, g, drag];
}

# Ground plane at y = 0 with restitution + friction.
fn step_with_ground(sys, dt, restitution, friction) {
    int i = 0; int n = 0; int positions = 0; int velocities = 0;
    if (restitution == nil) { restitution = 0.6; }
    if (friction == nil) { friction = 0.9; }
    sys = system_step(sys, dt);
    positions = sys[0];
    velocities = sys[1];
    n = len(positions) // 3;
    for (i = 0; i < n; ++i) {
        if (positions[i * 3 + 1] < 0.0) {
            positions[i * 3 + 1] = 0.0;
            if (velocities[i * 3 + 1] < 0.0) {
                velocities[i * 3 + 1] = 0.0 - velocities[i * 3 + 1] * restitution;
            }
            velocities[i * 3] = velocities[i * 3] * friction;
            velocities[i * 3 + 2] = velocities[i * 3 + 2] * friction;
        }
    }
    return [positions, velocities, sys[2], sys[3], sys[4], sys[5]];
}

# ---------------------------------------------------------------------------
# N-body gravitation (softened). bodies = [positions, velocities, masses].
# ---------------------------------------------------------------------------

fn n_body_step(bodies, gConst, softening, dt) {
    int acc = 0; int dx = 0; int dy = 0; int dz = 0; int fi = 0; int fj = 0; int i = 0; int invR = 0; int invR3 = 0; int j = 0; int masses = 0; int n = 0; int positions = 0; int r2 = 0; int s2 = 0; int velocities = 0;
    positions = bodies[0];
    velocities = bodies[1];
    masses = bodies[2];
    n = len(masses);
    acc = arrayx.zerosf(n * 3);
    s2 = softening * softening;
    for (i = 0; i < n; ++i) {
        for (j = i + 1; j < n; ++j) {
            dx = positions[j * 3] - positions[i * 3];
            dy = positions[j * 3 + 1] - positions[i * 3 + 1];
            dz = positions[j * 3 + 2] - positions[i * 3 + 2];
            r2 = dx * dx + dy * dy + dz * dz + s2;
            invR = 1.0 / sqrt(r2);
            invR3 = invR / r2;
            fi = gConst * masses[j] * invR3;
            fj = gConst * masses[i] * invR3;
            acc[i * 3] = acc[i * 3] + dx * fi;
            acc[i * 3 + 1] = acc[i * 3 + 1] + dy * fi;
            acc[i * 3 + 2] = acc[i * 3 + 2] + dz * fi;
            acc[j * 3] = acc[j * 3] - dx * fj;
            acc[j * 3 + 1] = acc[j * 3 + 1] - dy * fj;
            acc[j * 3 + 2] = acc[j * 3 + 2] - dz * fj;
        }
    }
    for (i = 0; i < n; ++i) {
        velocities[i * 3] = velocities[i * 3] + acc[i * 3] * dt;
        velocities[i * 3 + 1] = velocities[i * 3 + 1] + acc[i * 3 + 1] * dt;
        velocities[i * 3 + 2] = velocities[i * 3 + 2] + acc[i * 3 + 2] * dt;
        positions[i * 3] = positions[i * 3] + velocities[i * 3] * dt;
        positions[i * 3 + 1] = positions[i * 3 + 1] + velocities[i * 3 + 1] * dt;
        positions[i * 3 + 2] = positions[i * 3 + 2] + velocities[i * 3 + 2] * dt;
    }
    return [positions, velocities, masses];
}

fn kinetic_energy(velocities, masses) {
    int i = 0; int ke = 0; int n = 0; int v2 = 0;
    n = len(masses);
    ke = 0.0;
    for (i = 0; i < n; ++i) {
        v2 = velocities[i * 3] * velocities[i * 3] +
             velocities[i * 3 + 1] * velocities[i * 3 + 1] +
             velocities[i * 3 + 2] * velocities[i * 3 + 2];
        ke = ke + 0.5 * masses[i] * v2;
    }
    return ke;
}

fn momentum(velocities, masses) {
    int i = 0; int p = 0;
    p = [0.0, 0.0, 0.0];
    for (i = 0; i < len(masses); ++i) {
        p[0] = p[0] + masses[i] * velocities[i * 3];
        p[1] = p[1] + masses[i] * velocities[i * 3 + 1];
        p[2] = p[2] + masses[i] * velocities[i * 3 + 2];
    }
    return p;
}

# ---------------------------------------------------------------------------
# Collision helpers
# ---------------------------------------------------------------------------

# Elastic-ish sphere/sphere impulse. a/b = [pos, vel, radius, mass].
# Returns [aVel, bVel] (velocities after contact) or nil if not touching.
fn sphere_sphere(a, b, restitution) {
    int d = 0; int dist = 0; int jimp = 0; int ma = 0; int mb = 0; int n = 0; int pa = 0; int pb = 0; int ra = 0; int rb = 0; int rv = 0; int va = 0; int va2 = 0; int vb = 0; int vb2 = 0; int velN = 0;
    if (restitution == nil) { restitution = 1.0; }
    pa = a[0]; va = a[1]; ra = a[2]; ma = a[3];
    pb = b[0]; vb = b[1]; rb = b[2]; mb = b[3];
    d = vector.vsub(pb, pa);
    dist = vector.vnorm(d);
    if (dist > ra + rb || dist < 1.0e-9) { return nil; }
    n = vector.vscale(d, 1.0 / dist);
    rv = vector.vsub(vb, va);
    velN = vector.vdot(rv, n);
    if (velN > 0.0) { return nil; }           # separating already
    jimp = 0.0 - (1.0 + restitution) * velN / (1.0 / ma + 1.0 / mb);
    va2 = vector.vsub(va, vector.vscale(n, jimp / ma));
    vb2 = vector.vadd(vb, vector.vscale(n, jimp / mb));
    return [va2, vb2];
}

# Sphere vs axis-aligned box: returns [hit, pushedPos, reflectedVel].
fn sphere_aabb(pos, vel, radius, boxMin, boxMax, restitution) {
    int c = 0; int cl = 0; int d = 0; int dist = 0; int n = 0; int newPos = 0; int newVel = 0; int velN = 0;
    if (restitution == nil) { restitution = 0.5; }
    cl = [];
    for (c = 0; c < 3; ++c) {
        cl[c] = mathx.clampf(pos[c], boxMin[c], boxMax[c]);
    }
    d = vector.vsub(pos, cl);
    dist = vector.vnorm(d);
    if (dist > radius || dist < 1.0e-12) { return [false, pos, vel]; }
    n = vector.vscale(d, 1.0 / dist);
    newPos = vector.vadd(cl, vector.vscale(n, radius));
    velN = vector.vdot(vel, n);
    newVel = vel;
    if (velN < 0.0) {
        newVel = vector.vsub(vel, vector.vscale(n, (1.0 + restitution) * velN));
    }
    return [true, newPos, newVel];
}

# ---------------------------------------------------------------------------
# Cloth: (nx+1)x(ny+1) grid in the XZ plane at height y0, structural + shear
# springs, top row pinned if pinTop. Returns the ready system.
# ---------------------------------------------------------------------------

fn make_cloth(nx, ny, size, y0, pinTop, stiffness, damping) {
    int i = 0; int idx = 0; int j = 0; int m = 0; int p = 0; int r = 0; int step = 0; int stride = 0; int sys = 0;
    if (stiffness == nil) { stiffness = 60.0; }
    if (damping == nil) { damping = 0.5; }
    sys = make_system([0.0, -9.81, 0.0], 0.05);
    step = size / nx;
    for (j = 0; j <= ny; ++j) {
        for (i = 0; i <= nx; ++i) {
            m = 0.1;
            if (pinTop && j == 0) { m = 0.0; }
            p = [i * step - size / 2.0, y0, j * step];
            r = add_particle(sys, p, [0.0, 0.0, 0.0], m);
            sys = r[0];
        }
    }
    stride = nx + 1;
    for (j = 0; j <= ny; ++j) {
        for (i = 0; i <= nx; ++i) {
            idx = j * stride + i;
            if (i < nx) { sys = add_spring(sys, idx, idx + 1, stiffness, damping); }
            if (j < ny) { sys = add_spring(sys, idx, idx + stride, stiffness, damping); }
            if (i < nx && j < ny) {
                sys = add_spring(sys, idx, idx + stride + 1, stiffness * 0.6, damping);
                sys = add_spring(sys, idx + 1, idx + stride, stiffness * 0.6, damping);
            }
        }
    }
    return sys;
}
