# UV unwrapping & texturing: primitives, projections, seams, atlas, OBJ export.
import mesh;
import uvmap;
import texture;
import image;
import strx;
int atlas = 0;
int back = 0;
int boxed = 0;
int cube = 0;
int hi = 0;
int lo = 0;
int sphere = 0;
int tex = 0;
int ub = 0;
int unwrapped = 0;
# emx-scope-safe

print("== emeraldx: meshes, UV wrapping & texturing ==");
# --- primitives
sphere = mesh.icosphere(1.0, 2);
print("icosphere         ", mesh.vertex_count(sphere), " verts ",
      mesh.face_count(sphere), " faces, area ",
      strx.fmt_f(mesh.surface_area(sphere), 3), " (4pi=12.566)");

# --- spherical unwrap with seam fixing
unwrapped = uvmap.spherical_project(sphere);
print("spherical unwrap  ", mesh.vertex_count(sphere), " -> ",
      mesh.vertex_count(unwrapped), " verts (seam duplicates)");
ub = uvmap.uv_bounds(unwrapped);
lo = ub[0]; hi = ub[1];
print("uv range          u ", strx.fmt_f(lo[0],2), "..", strx.fmt_f(hi[0],2),
      "  v ", strx.fmt_f(lo[1],2), "..", strx.fmt_f(hi[1],2));

# --- box projection on a cube; overlap-free lightmap atlas
cube = mesh.cube(2.0);
boxed = uvmap.box_project(cube);
atlas = uvmap.atlas_pack_faces(cube, 0.08);
print("atlas area used   ", strx.fmt_f(uvmap.uv_area_used(atlas), 3), " of 1.0 (overlap-free)");

# --- texture to inspect the unwrap + OBJ export
tex = uvmap.checker_test_pattern(64, 8);
image.save(tex, "examples/out/uv_checker.ppm");
mesh.save_obj(unwrapped, "examples/out/sphere_unwrapped.obj");
mesh.save_obj(boxed, "examples/out/cube_boxmapped.obj");
back = mesh.load_obj("examples/out/sphere_unwrapped.obj");
print("obj roundtrip     ", mesh.vertex_count(back), " verts, area ",
      strx.fmt_f(mesh.surface_area(back), 3));
print("wrote examples/out/sphere_unwrapped.obj, cube_boxmapped.obj, uv_checker.ppm");
