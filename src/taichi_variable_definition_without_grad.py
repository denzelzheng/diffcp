from src.initialization import *

parameter_verbose_mode()

garment_E, garment_nu = scalar(), scalar()
shearing_stiffness, contact_stiffness = scalar(), scalar()
ti.root.place(garment_E, garment_nu)
ti.root.place(shearing_stiffness, contact_stiffness)

grid_v_out = vec()
ti.root.dense(ti.ijk, n_grid).place(grid_v_out)

operator_x = ti.Vector.field(3, dtype=ti.f32, shape=n_operator_particles)
operator_v = ti.Vector.field(3, dtype=ti.f32, shape=n_operator_particles)
operator_C = ti.Matrix.field(3, 3, dtype=real, shape=n_operator_particles)

cube_x = ti.Vector.field(3, dtype=ti.f32, shape=n_cube_particles)
cube_v = ti.Vector.field(3, dtype=ti.f32, shape=n_cube_particles)
cube_C = ti.Matrix.field(3, 3, dtype=real, shape=n_cube_particles)

grasp_tracking_seq = vec()
ti.root.dense(ti.k, n_actions).dense(ti.l, grasp_point_num).place(grasp_tracking_seq)

target_pos = ti.Vector.field(3, dtype=ti.f32, shape=n_target_particles)
lagrangian_mesh_element = ti.Vector.field(3, dtype=ti.i32, shape=n_quadrature_particles)

vertex_x = ti.Vector.field(3, dtype=ti.f32, shape=n_vertex_particles)
vertex_v = ti.Vector.field(3, dtype=ti.f32, shape=n_vertex_particles)
vertex_C = ti.Matrix.field(3, 3, dtype=real, shape=n_vertex_particles)
quadrature_x = ti.Vector.field(3, dtype=ti.f32, shape=n_quadrature_particles)
quadrature_v = ti.Vector.field(3, dtype=ti.f32, shape=n_quadrature_particles)
quadrature_C = ti.Matrix.field(3, 3, dtype=real, shape=n_quadrature_particles)
quadrature_F = ti.Matrix.field(3, 3, dtype=real, shape=n_quadrature_particles)
quadrature_d = ti.Matrix.field(3, 3, dtype=real, shape=n_quadrature_particles)

initial_operator_x = ti.Vector.field(3, dtype=ti.f32, shape=n_operator_particles)
initial_vertex_x = ti.Vector.field(3, dtype=ti.f32, shape=n_vertex_particles)
initial_cube_x = ti.Vector.field(3, dtype=ti.f32, shape=n_cube_particles)
tmp_operator_x = ti.Vector.field(3, dtype=ti.f32, shape=n_operator_particles)
tmp_vertex_x = ti.Vector.field(3, dtype=ti.f32, shape=n_vertex_particles)
tmp_cube_x = ti.Vector.field(3, dtype=ti.f32, shape=n_cube_particles)
initial_quadrature_d_inv = ti.Matrix.field(3, 3, dtype=real, shape=n_quadrature_particles)
loss = scalar()
ti.root.place(loss)
grid_v = ti.Vector.field(3, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))  # grid node momentum/velocity
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid, n_grid))  # grid node mass
inplane_F_tmp = ti.Matrix.field(2, 2, dtype=real, shape=n_quadrature_particles)
inplane_U = ti.Matrix.field(2, 2, dtype=real, shape=n_quadrature_particles)
inplane_V = ti.Matrix.field(2, 2, dtype=real, shape=n_quadrature_particles)
inplane_sig = ti.Matrix.field(2, 2, dtype=real, shape=n_quadrature_particles)
distances = ti.field(dtype=ti.f32, shape=(n_vertex_particles, n_target_particles))
