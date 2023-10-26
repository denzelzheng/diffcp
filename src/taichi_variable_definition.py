from src.initialization import *

parameter_verbose_mode()

garment_E, garment_nu = scalar(), scalar()
shearing_stiffness, contact_stiffness = scalar(), scalar()
ti.root.place(garment_E, garment_nu)
ti.root.place(shearing_stiffness, contact_stiffness)

learning_rate0 = scalar()
learning_rate1 = scalar()
learning_rate2 = scalar()
learning_rate3 = scalar()
ti.root.place(learning_rate0, learning_rate1, learning_rate2, learning_rate3)

grid_v_out = vec()
operator_x, operator_v = vec(), vec()
operator_C = mat()
ti.root.dense(ti.ijk, n_grid).place(grid_v_out)
ti.root.dense(ti.k, max_steps).dense(ti.l, n_operator_particles).place(operator_x, operator_v, operator_C)

cube_x, cube_v = vec(), vec()
cube_C = mat()
ti.root.dense(ti.k, max_steps).dense(ti.l, n_cube_particles).place(cube_x, cube_v, cube_C)
grasp_tracking_seq = vec()
ti.root.dense(ti.k, n_actions).dense(ti.l, grasp_point_num).place(grasp_tracking_seq)

target_pos = ti.Vector.field(3, dtype=ti.f32, shape=n_target_particles)
lagrangian_mesh_element = ti.Vector.field(3, dtype=ti.i32, shape=n_quadrature_particles)
vertex_x, vertex_v, vertex_C = vec(), vec(), mat()
quadrature_x, quadrature_v, quadrature_C, quadrature_F, quadrature_d = vec(), vec(), mat(), mat(), mat()  # Lagrangian mesh element quadrature particles
ti.root.dense(ti.k, max_steps).dense(ti.l, n_vertex_particles).place(vertex_x, vertex_v, vertex_C)
ti.root.dense(ti.k, max_steps).dense(ti.l, n_quadrature_particles).place(quadrature_x, quadrature_v, quadrature_C,
                                                                         quadrature_F, quadrature_d)
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

ti.root.lazy_grad()

def parameter_verbose_mode():
    print("steps is", max_steps)
    print("n_vertex_particles", n_vertex_particles)
    print("quadrature_p_mass", garment_mass / n_vertex_particles)
    print("quadrature_p_vol", quadrature_p_vol)
    print("garment_p_mass", quadrature_p_mass)
    print("operator_p_vol", operator_p_vol)
    print("cube_p_vol", cube_p_vol)
    print("operator_p_mass", operator_p_mass)
    print("cube_p_mass", cube_p_mass)
    print("operator mass is", operator_p_mass * n_operator_particles)
    print("soft_body_model_vertices:", soft_body_model_vertices.shape)
    print("There are", n_actions, "frames will be used.")