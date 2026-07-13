# =============================================================================
# render.em - a small software rasterizer
#
# Pipeline: model -> view -> projection (mat4), perspective divide, viewport,
# back-face culling, z-buffered barycentric rasterization with perspective-
# correct interpolation of normals and UVs, Lambert diffuse + Blinn-Phong
# specular, optional texture lookup (texture.sample_uv).
#
#   target = render.make_target(w, h, bgColor)
#   target = render.draw_mesh(target, meshv, model, view, proj, mat, light)
#   render.save_target(target, "out.ppm")
#
# target = [img, zbuf]  (zbuf flat, +inf = empty)
# material = [baseColorRGB, textureImageOrNil, kAmbient, kDiffuse,
#             kSpecular, shininess]
# light = [directionTOWARDSlight(3, normalized), colorRGB]
# =============================================================================

import mathx;
import arrayx;
import vector;
import mat4;
import mesh;
import image;
import texture;
# emx-scope-safe

fn make_target(w, h, bg) {
    int d = 0; int i = 0; int img = 0; int zbuf = 0;
    if (bg == nil) { bg = [0.05, 0.06, 0.08]; }
    img = image.new_image(w, h, 3, 0.0);
    d = img[3];
    for (i = 0; i < w * h; ++i) {
        d[i * 3] = bg[0];
        d[i * 3 + 1] = bg[1];
        d[i * 3 + 2] = bg[2];
    }
    zbuf = arrayx.full(w * h, 1.0e30);
    return [[w, h, 3, d], zbuf];
}

fn save_target(target, path) {
    return image.save(target[0], path);
}

fn default_material(color) {
    if (color == nil) { color = [0.8, 0.8, 0.85]; }
    return [color, nil, 0.15, 0.75, 0.35, 24.0];
}

fn textured_material(tex) {
    return [[1.0, 1.0, 1.0], tex, 0.2, 0.8, 0.25, 16.0];
}

fn default_light() {
    return [vector.vnormalize([0.5, 0.8, 0.6]), [1.0, 1.0, 1.0]];
}

# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------

# Draw a mesh. Vertices are shaded per-pixel in WORLD space; the camera
# position (for speculars) is recovered from the inverse view matrix.
fn draw_mesh(target, m, model, view, proj, material, light) {
    int area = 0; int base = 0; int baseColor = 0; int cb = 0; int cg = 0; int clip = 0; int cr = 0; int cw = 0; int cx = 0; int cy = 0; int eye = 0; int f = 0; int faces = 0; int h = 0; int hasN = 0; int hasUV = 0; int hl = 0; int hx = 0; int hy = 0; int hz = 0; int i = 0; int i0 = 0; int i1 = 0; int i2 = 0; int img = 0; int invArea = 0; int invView = 0; int iw0 = 0; int iw1 = 0; int iw2 = 0; int ka = 0; int kd = 0; int ks = 0; int lightCol = 0; int lightDir = 0; int ln0 = 0; int ln1 = 0; int ln2 = 0; int lum = 0; int maxx = 0; int maxy = 0; int minx = 0; int miny = 0; int mv = 0; int mvp = 0; int n0t = 0; int n1t = 0; int n2t = 0; int ndcx = 0; int ndcy = 0; int ndh = 0; int ndl = 0; int nf = 0; int nl = 0; int normals = 0; int nrmMat = 0; int nv = 0; int nx = 0; int ny = 0; int nz = 0; int p = 0; int pix = 0; int positions = 0; int pwx = 0; int pwy = 0; int pwz = 0; int px2 = 0; int py = 0; int q0 = 0; int q1 = 0; int q2 = 0; int qs = 0; int shin = 0; int spec = 0; int sw = 0; int sx = 0; int sy = 0; int sz = 0; int tc = 0; int tex = 0; int u = 0; int uvs = 0; int v = 0; int vl = 0; int vxd = 0; int vyd = 0; int vzd = 0; int w = 0; int w0 = 0; int w1 = 0; int w2 = 0; int wp = 0; int wx = 0; int wy = 0; int wz = 0; int x0 = 0; int x1 = 0; int x2 = 0; int y0 = 0; int y1 = 0; int y2 = 0; int z = 0; int zbuf = 0; int zi = 0;
    img = target[0];
    zbuf = target[1];
    w = img[0];
    h = img[1];
    pix = img[3];

    positions = m[0];
    normals = m[1];
    uvs = m[2];
    faces = m[3];
    hasUV = len(uvs) > 0;
    hasN = len(normals) > 0;

    mv = mat4.mul(view, model);
    mvp = mat4.mul(proj, mv);
    nrmMat = mat4.normal_matrix(model);
    invView = mat4.inverse(view);
    eye = [invView[3], invView[7], invView[11]];

    baseColor = material[0];
    tex = material[1];
    ka = material[2];
    kd = material[3];
    ks = material[4];
    shin = material[5];
    lightDir = light[0];
    lightCol = light[1];

    nv = len(positions) // 3;
    # transform all vertices once
    sx = [];   # screen x
    sy = [];   # screen y
    sz = [];   # ndc z
    sw = [];   # clip w (for perspective correction)
    wx = [];   # world position
    wy = [];
    wz = [];
    for (i = 0; i < nv; ++i) {
        p = [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];
        wp = mat4.transform_point(model, p);
        wx[i] = wp[0];
        wy[i] = wp[1];
        wz[i] = wp[2];
        clip = mat4.transform_vec4(mvp, [p[0], p[1], p[2], 1.0]);
        cw = clip[3];
        if (abs(cw) < 1.0e-9) { cw = 1.0e-9; }
        ndcx = clip[0] / cw;
        ndcy = clip[1] / cw;
        sz[i] = clip[2] / cw;
        sw[i] = cw;
        sx[i] = (ndcx * 0.5 + 0.5) * (w - 1);
        sy[i] = (1.0 - (ndcy * 0.5 + 0.5)) * (h - 1);
    }

    nf = len(faces) // 3;
    for (f = 0; f < nf; ++f) {
        i0 = faces[f * 3];
        i1 = faces[f * 3 + 1];
        i2 = faces[f * 3 + 2];
        # clip: skip triangles with any vertex behind the camera
        if (sw[i0] <= 0.0 || sw[i1] <= 0.0 || sw[i2] <= 0.0) { continue; }
        x0 = sx[i0]; y0 = sy[i0];
        x1 = sx[i1]; y1 = sy[i1];
        x2 = sx[i2]; y2 = sy[i2];
        area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
        if (area >= 0.0) { continue; }        # back-face cull (CCW convention)
        minx = mathx.maxi(int(floor(mathx.minv(x0, mathx.minv(x1, x2)))), 0);
        maxx = mathx.mini(int(top(mathx.maxv(x0, mathx.maxv(x1, x2)))), w - 1);
        miny = mathx.maxi(int(floor(mathx.minv(y0, mathx.minv(y1, y2)))), 0);
        maxy = mathx.mini(int(top(mathx.maxv(y0, mathx.maxv(y1, y2)))), h - 1);
        if (minx > maxx || miny > maxy) { continue; }
        invArea = 1.0 / area;
        iw0 = 1.0 / sw[i0];
        iw1 = 1.0 / sw[i1];
        iw2 = 1.0 / sw[i2];
        for (py = miny; py <= maxy; ++py) {
            for (px2 = minx; px2 <= maxx; ++px2) {
                cx = px2 + 0.0;
                cy = py + 0.0;
                w0 = ((x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)) * invArea;
                w1 = ((x0 - x2) * (cy - y2) - (y0 - y2) * (cx - x2)) * invArea;
                w2 = 1.0 - w0 - w1;
                if (w0 < 0.0 || w1 < 0.0 || w2 < 0.0) { continue; }
                z = w0 * sz[i0] + w1 * sz[i1] + w2 * sz[i2];
                zi = py * w + px2;
                if (z >= zbuf[zi]) { continue; }
                zbuf[zi] = z;
                # perspective-correct weights
                q0 = w0 * iw0;
                q1 = w1 * iw1;
                q2 = w2 * iw2;
                qs = q0 + q1 + q2;
                q0 = q0 / qs;
                q1 = q1 / qs;
                q2 = q2 / qs;
                # world position
                pwx = q0 * wx[i0] + q1 * wx[i1] + q2 * wx[i2];
                pwy = q0 * wy[i0] + q1 * wy[i1] + q2 * wy[i2];
                pwz = q0 * wz[i0] + q1 * wz[i1] + q2 * wz[i2];
                # normal
                nx = 0.0; ny = 0.0; nz = 1.0;
                if (hasN) {
                    ln0 = [normals[i0 * 3], normals[i0 * 3 + 1], normals[i0 * 3 + 2]];
                    ln1 = [normals[i1 * 3], normals[i1 * 3 + 1], normals[i1 * 3 + 2]];
                    ln2 = [normals[i2 * 3], normals[i2 * 3 + 1], normals[i2 * 3 + 2]];
                    n0t = mat4.transform_dir(nrmMat, ln0);
                    n1t = mat4.transform_dir(nrmMat, ln1);
                    n2t = mat4.transform_dir(nrmMat, ln2);
                    nx = q0 * n0t[0] + q1 * n1t[0] + q2 * n2t[0];
                    ny = q0 * n0t[1] + q1 * n1t[1] + q2 * n2t[1];
                    nz = q0 * n0t[2] + q1 * n1t[2] + q2 * n2t[2];
                    nl = sqrt(nx * nx + ny * ny + nz * nz);
                    if (nl < 1.0e-9) { nl = 1.0; }
                    nx = nx / nl; ny = ny / nl; nz = nz / nl;
                }
                # base color (texture or flat)
                cr = baseColor[0];
                cg = baseColor[1];
                cb = baseColor[2];
                if (tex != nil && hasUV) {
                    u = q0 * uvs[i0 * 2] + q1 * uvs[i1 * 2] + q2 * uvs[i2 * 2];
                    v = q0 * uvs[i0 * 2 + 1] + q1 * uvs[i1 * 2 + 1] + q2 * uvs[i2 * 2 + 1];
                    tc = texture.sample_uv_rgb(tex, u, v);
                    cr = cr * tc[0];
                    cg = cg * tc[1];
                    cb = cb * tc[2];
                }
                # lambert
                ndl = nx * lightDir[0] + ny * lightDir[1] + nz * lightDir[2];
                if (ndl < 0.0) { ndl = 0.0; }
                # blinn-phong
                vxd = eye[0] - pwx;
                vyd = eye[1] - pwy;
                vzd = eye[2] - pwz;
                vl = sqrt(vxd * vxd + vyd * vyd + vzd * vzd);
                if (vl < 1.0e-9) { vl = 1.0; }
                vxd = vxd / vl; vyd = vyd / vl; vzd = vzd / vl;
                hx = vxd + lightDir[0];
                hy = vyd + lightDir[1];
                hz = vzd + lightDir[2];
                hl = sqrt(hx * hx + hy * hy + hz * hz);
                if (hl < 1.0e-9) { hl = 1.0; }
                ndh = (nx * hx + ny * hy + nz * hz) / hl;
                if (ndh < 0.0) { ndh = 0.0; }
                spec = ks * (ndh ** shin);
                lum = ka + kd * ndl;
                base = zi * 3;
                pix[base] = image.clamp01(cr * lum * lightCol[0] + spec * lightCol[0]);
                pix[base + 1] = image.clamp01(cg * lum * lightCol[1] + spec * lightCol[1]);
                pix[base + 2] = image.clamp01(cb * lum * lightCol[2] + spec * lightCol[2]);
            }
        }
    }
    return [[w, h, 3, pix], zbuf];
}

# Convenience: orbiting camera looking at the origin.
fn orbit_camera(distance, yaw, pitch, fovDeg, aspect) {
    int cx = 0; int cy = 0; int cz = 0; int proj = 0; int view = 0;
    cx = distance * cos(pitch) * sin(yaw);
    cy = distance * sin(pitch);
    cz = distance * cos(pitch) * cos(yaw);
    view = mat4.look_at([cx, cy, cz], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    proj = mat4.perspective(fovDeg, aspect, 0.1, 100.0);
    return [view, proj];
}

# Render the z-buffer as a gray image (near = bright) for debugging.
fn depth_image(target) {
    int d = 0; int h = 0; int hi = 0; int i = 0; int img = 0; int lo = 0; int w = 0; int z = 0; int zbuf = 0;
    img = target[0];
    zbuf = target[1];
    w = img[0];
    h = img[1];
    lo = 1.0e30;
    hi = -1.0e30;
    for (i = 0; i < w * h; ++i) {
        z = zbuf[i];
        if (z < 1.0e29) {
            if (z < lo) { lo = z; }
            if (z > hi) { hi = z; }
        }
    }
    if (hi <= lo) { hi = lo + 1.0; }
    d = [];
    for (i = 0; i < w * h; ++i) {
        z = zbuf[i];
        if (z > 1.0e29) { d[i] = 0.0; }
        else { d[i] = 1.0 - (z - lo) / (hi - lo); }
    }
    return [w, h, 1, d];
}
