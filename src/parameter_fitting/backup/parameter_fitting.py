
import taichi as ti
import numpy as np
import random
import taichi_glsl as ts
import math
import sys
import os
import open3d as o3d
import json
import time


real = ti.f32
index = lambda: ti.field(dtype=ti.i32)
scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(3, dtype=real)
vec_2d = lambda: ti.Vector.field(2, dtype=real)
mat = lambda: ti.Matrix.field(3, 3, dtype=real)

ti.reset()
ti.init(arch=ti.cuda, device_memory_GB=21.3,
        kernel_profiler=True, default_fp=real)

dim = 3
bound = 3
gravity = -9.81
print()

current_dir = os.path.abspath(os.path.dirname(__file__))
arguments_file = open(current_dir + '/parameter_fitting_arguments.json')
arguments = json.load(arguments_file)
operator_parameters = arguments['operator_parameters']
IFS = arguments['IFS']
dt = arguments['dt']
n_grid = arguments['n_grid']
garment_scaling = arguments['garment_scaling']
initial_frame = arguments['initial_frame']
end_frame = arguments['end_frame']
operator_edge_length = operator_parameters['operator_edge_length']
n_operator_particles = operator_parameters['n_operator_particles']
operator_mu = operator_parameters['operator_mu']
operator_la = operator_parameters['operator_la']
garment_parameters = arguments['garment_parameters']
upper_bounds = garment_parameters['constitutive_parameter_upper_bounds']
lower_bounds = garment_parameters['constitutive_parameter_lower_bounds']
data_dir = arguments['data_dir']
result_dir = arguments['result_dir']
number_of_triangles = garment_parameters['number_of_triangles']
operator_mass = operator_parameters['operator_mass']
rigid_cube_parameters = arguments['rigid_cube_parameters']
cube_mu = rigid_cube_parameters['cube_mu']
cube_la = rigid_cube_parameters['cube_la']
cube_mass = rigid_cube_parameters['cube_mass']
rigid_cube_side_length = rigid_cube_parameters['rigid_cube_side_length']
grad_update_IFS = arguments['grad_update_IFS']
n_sample_target_points = arguments['n_sample_target_points']

E_lower_bound = lower_bounds['E']
nu_lower_bound = lower_bounds['nu']
contact_stiffness_lower_bound = lower_bounds['contact_stiffness']
shearing_stiffness_lower_bound = lower_bounds['shearing_stiffness']

n_rigid_cube_particles = rigid_cube_parameters['n_rigid_cube_particles']

E_upper_bound = upper_bounds['E']
nu_upper_bound = upper_bounds['nu']
contact_stiffness_upper_bound = upper_bounds['contact_stiffness']
shearing_stiffness_upper_bound = upper_bounds['shearing_stiffness']

garment_mass = garment_parameters['garment_mass']
garment_mass = garment_mass * (garment_scaling ** 3)


data_dir = current_dir + data_dir
result_file_path = current_dir + result_dir + '/fit_parameter.json'
cloth_file = data_dir + '/world_mesh.ply'
mark_file = data_dir + '/world_mark.npy'
world_target_path = data_dir + '/target_sample.npy'
target_pcd_array = np.load(world_target_path)
mark_pos = np.load(mark_file)
grasp_seq = np.load(data_dir + "/traj_world_array.npy")


dx, inv_dx = 1 / n_grid, float(n_grid)
initial_step = 1

n_grad_updates = target_pcd_array.shape[0]
n_target_particles = target_pcd_array.shape[1]

random_points = np.random.rand(n_rigid_cube_particles, 3) * rigid_cube_side_length - rigid_cube_side_length / 2
random_points[:, 0] += mark_pos[0]
random_points[:, 1] += mark_pos[1] - rigid_cube_side_length/2
random_points[:, 2] += mark_pos[2]
cube_model = random_points

frame_range = grasp_seq.shape[0]
print("there are", frame_range, "frames in data")
if end_frame >= frame_range or end_frame == -1:
    end_frame = frame_range - 1
if initial_frame > end_frame:
    initial_frame = end_frame - 1
print("initial_frame is", initial_frame)
print("end_frame is", end_frame)

grasp_seq = grasp_seq[initial_frame:end_frame + 1]




frame_range = end_frame - initial_frame + 1
grasp_point_num = grasp_seq[0].shape[0]
soft_body_mesh = o3d.io.read_triangle_mesh(cloth_file)


soft_body_model_vertices = np.asarray(soft_body_mesh.vertices)
soft_body_model_vertices = soft_body_model_vertices * garment_scaling
soft_body_mesh.vertices = o3d.utility.Vector3dVector(soft_body_model_vertices)
soft_body_model_triangles = np.asarray(soft_body_mesh.triangles)
grasp_seq = grasp_seq * garment_scaling
cube_model = cube_model * garment_scaling
target_pcd_array = target_pcd_array * garment_scaling
garment_surface_area = soft_body_mesh.get_surface_area()
n_vertex_particles = soft_body_model_vertices.shape[0]
n_quadrature_particles = soft_body_model_triangles.shape[0]
n_operator_particles = int(n_operator_particles / grasp_point_num) * grasp_point_num
n_cube_particles = cube_model.shape[0]


garment_rho = garment_mass / (garment_surface_area * dx)
operator_edge_length *= garment_scaling
operator_volume = operator_edge_length ** 3
rigid_cube_side_length *= garment_scaling
cube_p_vol = rigid_cube_side_length ** 3 / n_cube_particles
cube_p_mass = cube_mass / n_cube_particles
operator_p_vol = operator_volume / n_operator_particles
operator_p_mass = operator_mass / n_operator_particles


quadrature_p_vol = 3 * garment_surface_area * dx / n_quadrature_particles
vertex_p_vol = garment_surface_area * dx / n_vertex_particles
quadrature_p_mass = garment_mass / n_vertex_particles
vertex_p_mass = vertex_p_vol * garment_rho


lower_bound = dx * (bound - 1)
upper_bound = (1 - dx * (bound - 1))
points = [
    [lower_bound, lower_bound, lower_bound],
    [upper_bound, lower_bound, lower_bound],
    [lower_bound, upper_bound, lower_bound],
    [upper_bound, upper_bound, lower_bound],
    [lower_bound, lower_bound, upper_bound],
    [upper_bound, lower_bound, upper_bound],
    [lower_bound, upper_bound, upper_bound],
    [upper_bound, upper_bound, upper_bound],
]
lines = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 3],
    [4, 5],
    [4, 6],
    [5, 7],
    [6, 7],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
]
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)


operator_model_without_position = list()
grasper_group_size = int(n_operator_particles / grasp_point_num)
for i in range(grasper_group_size):
    operator_model_without_position.append([(random.random() - 0.5) * operator_edge_length,
                                            (random.random() - 0.5) * operator_edge_length,
                                            (random.random() - 0.5) * operator_edge_length])
operator_model_without_position = np.array(operator_model_without_position)


model_position = [0, (np.amax(soft_body_model_vertices, axis=0) - np.amin(soft_body_model_vertices, axis=0))[1] / 2 +
                  np.amin(soft_body_model_vertices, axis=0)[1], 0]
soft_body_model_bottom = [0, np.amin(soft_body_model_vertices, axis=0)[1], 0]



scene = np.concatenate((soft_body_model_vertices, grasp_seq.reshape(-1, 3)))
scene_center = (np.amax(scene, axis=0) - np.amin(scene, axis=0)) / 2 + np.amin(scene, axis=0)
scene_bottom_center = np.array([scene_center[0], 0, scene_center[2]])
new_initial_scene_center = np.array([0.5, 2 * dx, 0.5])
soft_body_model_vertices = soft_body_model_vertices + new_initial_scene_center - scene_bottom_center
grasp_point_pos = grasp_seq + new_initial_scene_center - scene_bottom_center
cube_model = cube_model + new_initial_scene_center - scene_bottom_center
target_pcd_array = target_pcd_array + new_initial_scene_center - scene_bottom_center
grasp_point_pos_pcd = o3d.geometry.PointCloud()
grasp_point_pos_pcd.points = o3d.utility.Vector3dVector(grasp_point_pos.reshape(-1, 3))
grasp_point_pos_pcd.paint_uniform_color([x / 255 for x in [255, 0, 0]])
cube_pcd_show = o3d.geometry.PointCloud()
cube_pcd_show.points = o3d.utility.Vector3dVector(np.array(cube_model))
soft_body_mesh.vertices = o3d.utility.Vector3dVector(soft_body_model_vertices)
target_pcd_show = o3d.geometry.PointCloud()
target_pcd_show.points = o3d.utility.Vector3dVector(np.array(target_pcd_array[4]))
o3d.visualization.draw_geometries([target_pcd_show, line_set, grasp_point_pos_pcd, soft_body_mesh, cube_pcd_show])


pos_series = list()
operator_pos_series = list()
target_pos_series = list()
operator_target_pos_series = list()
min_loss, best_E, best_nu = 0, 0, 0



max_steps = int(grad_update_IFS / dt) + 1

garment_E, garment_nu = scalar(), scalar()
shearing_stiffness, contact_stiffness = scalar(), scalar()
ti.root.place(garment_E, garment_nu)
ti.root.place(shearing_stiffness, contact_stiffness)


# lr fields
learning_rate0 = scalar()
learning_rate1 = scalar()
learning_rate2 = scalar()
learning_rate3 = scalar()
learning_rate4 = scalar()
learning_rate5 = scalar()
ti.root.place(learning_rate0, learning_rate1, learning_rate2)
ti.root.place(learning_rate3, learning_rate4, learning_rate5)

operator_center = vec()
grid_v_out = vec()
operator_x, operator_v = vec(), vec()
operator_C = mat()
initial_operator_x = ti.Vector.field(3, dtype=ti.f32, shape=n_operator_particles)
ti.root.dense(ti.ijk, n_grid).place(grid_v_out)
ti.root.dense(ti.k, max_steps).dense(ti.l, n_operator_particles).place(operator_center, operator_x, operator_v, operator_C)


cube_x, cube_v = vec(), vec()
cube_C, cube_F = mat(), mat()
ti.root.dense(ti.k, max_steps).dense(ti.l, n_cube_particles).place(cube_x, cube_v, cube_C, cube_F)

grasp_tracking_seq = vec()
ti.root.dense(ti.k, frame_range).dense(ti.l, grasp_point_num).place(grasp_tracking_seq)

# fields in anisotropic simulation
target_pos = ti.Vector.field(3, dtype=ti.f32, shape=n_target_particles)
initial_vertex_x = ti.Vector.field(3, dtype=ti.f32, shape=n_vertex_particles)
initial_cube_x = ti.Vector.field(3, dtype=ti.f32, shape=n_cube_particles)
lagrangian_mesh_element = ti.Vector.field(3, dtype=ti.i32, shape=n_quadrature_particles)
vertex_x, vertex_v, vertex_C = vec(), vec(), mat()
quadrature_x, quadrature_v, quadrature_C, quadrature_F, quadrature_d = vec(), vec(), mat(), mat(), mat()  # Lagrangian mesh element quadrature particles
ti.root.dense(ti.k, max_steps).dense(ti.l, n_vertex_particles).place(vertex_x, vertex_v, vertex_C)
ti.root.dense(ti.k, max_steps).dense(ti.l, n_quadrature_particles).place(quadrature_x, quadrature_v, quadrature_C,
                                                                         quadrature_F, quadrature_d)
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
    print("There are", frame_range, "frames will be used.")
parameter_verbose_mode()

@ti.kernel
def clear_distances():
    for p in range(n_vertex_particles):
        for q in range(n_target_particles):
            distances[p, q] = 0

@ti.kernel
def clear_grid():
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0, 0, 0]
        grid_m[i, j, k] = 0
        grid_v.grad[i, j, k] = [0, 0, 0]
        grid_m.grad[i, j, k] = 0
        grid_v_out.grad[i, j, k] = [0, 0, 0]


def zero_vec():
    return [0.0, 0.0, 0.0]


def zero_matrix():
    return [zero_vec(), zero_vec(), zero_vec()]

@ti.kernel
def compute_inplane_F_tmp(f: ti.i32):
    for p in range(0, n_quadrature_particles):
        Q, R = QR_decomposition_3d(quadrature_F[f, p])
        inplane_F_tmp[p] = ti.Matrix([[R[0, 0], R[0, 1]], [R[1, 0], R[1, 1]]])

@ti.kernel
def clear_inplane_SVD_grad():
    zero = ti.Matrix.zero(real, 2, 2)
    for i in range(0, n_quadrature_particles):
        inplane_U.grad[i] = zero
        inplane_sig.grad[i] = zero
        inplane_V.grad[i] = zero
        inplane_F_tmp.grad[i] = zero



@ti.kernel
def inplane_svd():
    for p in range(0, n_quadrature_particles):
        inplane_U[p], inplane_sig[p], inplane_V[p] = ti.svd(inplane_F_tmp[p])


@ti.func
def clamp(a):
    if a >= 0:
        a = max(a, 1e-6)
    else:
        a = min(a, -1e-6)
    return a


@ti.func
def backward_svd(gu, gsigma, gv, u, sig, v):
    # https://github.com/pytorch/pytorch/blob/ab0a04dc9c8b84d4a03412f1c21a6c4a2cefd36c/tools/autograd/templates/Functions.cpp
    vt = v.transpose()
    ut = u.transpose()
    sigma_term = u @ gsigma @ vt

    s = ti.Vector.zero(real, 3)
    s = ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]]) ** 2
    F = ti.Matrix.zero(real, 3, 3)
    for i, j in ti.static(ti.ndrange(3, 3)):
        if i == j:
            F[i, j] = 0
        else:
            F[i, j] = 1. / clamp(s[j] - s[i])
    u_term = u @ ((F * (ut @ gu - gu.transpose() @ u)) @ sig) @ vt
    v_term = u @ (sig @ ((F * (vt @ gv - gv.transpose() @ v)) @ vt))
    return u_term + v_term + sigma_term

@ti.func
def backward_inplane_svd(gu, gsigma, gv, u, sig, v):
    # https://github.com/pytorch/pytorch/blob/ab0a04dc9c8b84d4a03412f1c21a6c4a2cefd36c/tools/autograd/templates/Functions.cpp
    vt = v.transpose()
    ut = u.transpose()
    sigma_term = u @ gsigma @ vt

    s = ti.Vector.zero(real, 2)
    s = ti.Vector([sig[0, 0], sig[1, 1]]) ** 2
    F = ti.Matrix.zero(real, 2, 2)
    for i, j in ti.static(ti.ndrange(2, 2)):
        if i == j:
            F[i, j] = 0
        else:
            F[i, j] = 1. / clamp(s[j] - s[i])
    u_term = u @ ((F * (ut @ gu - gu.transpose() @ u)) @ sig) @ vt
    v_term = u @ (sig @ ((F * (vt @ gv - gv.transpose() @ v)) @ vt))
    return u_term + v_term + sigma_term

@ti.kernel
def inplane_svd_grad():
    for p in range(0, n_quadrature_particles):
        inplane_F_tmp.grad[p] += backward_inplane_svd(inplane_U.grad[p], inplane_sig.grad[p], inplane_V.grad[p], inplane_U[p],
                                              inplane_sig[p], inplane_V[p])


@ti.func
def make_matrix_from_diag(d):
    return ti.Matrix([[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]], dt=real)


@ti.func
def norm(x, eps=1e-8):
    return ti.sqrt(x.dot(x) + eps)


@ti.func
def compute_inplane_P_hat(sig, mu, la):
    epsilon = ti.Vector([ti.log(sig[0, 0]), ti.log(sig[1, 1])])
    sum_log = epsilon[0] + epsilon[1]
    psi_0 = (2 * mu * epsilon[0] + la * sum_log) / sig[0, 0]
    psi_1 = (2 * mu * epsilon[1] + la * sum_log) / sig[1, 1]
    P_hat = ti.Vector([psi_0, psi_1])
    return P_hat


@ti.func
def compute_P_hat(sig, mu, la):
    epsilon = ti.Vector([ti.log(sig[0, 0]), ti.log(sig[1, 1]), ti.log(sig[2, 2])])
    sum_log = epsilon[0] + epsilon[1] + epsilon[2]
    psi_0 = (2 * mu * epsilon[0] + la * sum_log) / sig[0, 0]
    psi_1 = (2 * mu * epsilon[1] + la * sum_log) / sig[1, 1]
    psi_2 = (2 * mu * epsilon[2] + la * sum_log) / sig[2, 2]
    P_hat = ti.Vector([psi_0, psi_1, psi_2])
    return P_hat


@ti.func
def QR_decomposition_3d(F_mat):
    col_1 = ti.Vector([F_mat[0, 0], F_mat[1, 0], F_mat[2, 0]])
    col_2 = ti.Vector([F_mat[0, 1], F_mat[1, 1], F_mat[2, 1]])
    col_3 = ti.Vector([F_mat[0, 2], F_mat[1, 2], F_mat[2, 2]])
    q_col_1 = col_1 / (col_1.norm(1e-9))
    q_col_2_hat = col_2 - (col_2.dot(q_col_1) * q_col_1)
    q_col_2 = q_col_2_hat / (q_col_2_hat.norm(1e-9))
    q_col_3_hat = col_3 - (col_3.dot(q_col_1) * q_col_1) - (col_3.dot(q_col_2) * q_col_2)
    q_col_3 = q_col_3_hat / (q_col_3_hat.norm(1e-9))
    Q = ti.Matrix.cols([q_col_1, q_col_2, q_col_3])
    R = Q.inverse() @ F_mat
    return Q, R


@ti.kernel
def p2g(f: ti.i32, grad_update: ti.i32):
    for p in range(0, n_cube_particles):
        base = ti.cast(cube_x[f, p] * inv_dx - 0.5, ti.i32)
        fx = cube_x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        affine = cube_p_mass * cube_C[f, p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    offset = ti.Vector([i, j, k])
                    dpos = (ti.cast(ti.Vector([i, j, k]), real) - fx) * dx
                    weight = w[i][0] * w[j][1] * w[k][2]
                    ti.atomic_add(grid_v[base + offset],
                                  weight * (affine @ dpos))
                    ti.atomic_add(grid_m[base + offset], weight * cube_p_mass)



    for p in range(0, n_operator_particles):
        base = ti.cast(operator_x[f, p] * inv_dx - 0.5, ti.i32)
        fx = operator_x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        affine = operator_p_mass * operator_C[f, p]

        t = f * dt
        frame = int(t / IFS) + int(grad_update_IFS / IFS) * grad_update
        grasp_id = int(p / grasper_group_size)
        operator_temp_v = ti.Vector([0.0, 0.0, 0.0])
        if frame + 1 < frame_range:
            operator_temp_v = (grasp_tracking_seq[frame + 1, grasp_id] - grasp_tracking_seq[frame, grasp_id]) / IFS
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    offset = ti.Vector([i, j, k])
                    dpos = (ti.cast(ti.Vector([i, j, k]), real) - fx) * dx
                    weight = w[i][0] * w[j][1] * w[k][2]
                    ti.atomic_add(grid_v[base + offset],
                                  weight * (operator_p_mass * operator_temp_v + affine @ dpos))
                    ti.atomic_add(grid_m[base + offset], weight * operator_p_mass)



    for p in range(0, n_vertex_particles):

        base = ti.cast(vertex_x[f, p] * inv_dx - 0.5, ti.i32)
        fx = vertex_x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        affine = vertex_p_mass * vertex_C[f, p]
        temp_v = vertex_v[f, p]
        temp_v = vertex_v[f, p] + [0, dt * garment_scaling * gravity, 0]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    offset = ti.Vector([i, j, k])
                    dpos = (ti.cast(ti.Vector([i, j, k]), real) - fx) * dx
                    weight = w[i][0] * w[j][1] * w[k][2]
                    ti.atomic_add(grid_v[base + offset],
                                  weight * (vertex_p_mass * temp_v + affine @ dpos))
                    ti.atomic_add(grid_m[base + offset], weight * vertex_p_mass)

    for p in range(0, n_quadrature_particles):

        base = ti.cast(quadrature_x[f, p] * inv_dx - 0.5, ti.i32)
        fx = quadrature_x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        x_jacobian_w = ti.Matrix.rows([fx - 1.5, 2 * (1.0 - fx), fx - 0.5]) * inv_dx
        affine = quadrature_C[f, p] * quadrature_p_mass
        temp_v = quadrature_v[f, p]

        v1 = lagrangian_mesh_element[p][0]
        v2 = lagrangian_mesh_element[p][1]
        v3 = lagrangian_mesh_element[p][2]
        v1_base = ti.cast(vertex_x[f, v1] * inv_dx - 0.5, ti.i32)
        v1_fx = vertex_x[f, v1] * inv_dx - ti.cast(v1_base, ti.i32)
        v1_w = [0.5 * (1.5 - v1_fx) ** 2, 0.75 - (v1_fx - 1) ** 2, 0.5 * (v1_fx - 0.5) ** 2]
        v2_base = ti.cast(vertex_x[f, v2] * inv_dx - 0.5, ti.i32)
        v2_fx = vertex_x[f, v2] * inv_dx - ti.cast(v2_base, ti.i32)
        v2_w = [0.5 * (1.5 - v2_fx) ** 2, 0.75 - (v2_fx - 1) ** 2, 0.5 * (v2_fx - 0.5) ** 2]
        v3_base = ti.cast(vertex_x[f, v3] * inv_dx - 0.5, ti.i32)
        v3_fx = vertex_x[f, v3] * inv_dx - ti.cast(v3_base, ti.i32)
        v3_w = [0.5 * (1.5 - v3_fx) ** 2, 0.75 - (v3_fx - 1) ** 2, 0.5 * (v3_fx - 0.5) ** 2]

        Q, R = QR_decomposition_3d(quadrature_F[f, p])
        A = ti.Matrix(zero_matrix())
        A[0, 0] = shearing_stiffness[None] * R[0, 2] * R[0, 2] + garment_E[None] * R[0, 0] * (R[0, 0] - 1)
        A[0, 0] = shearing_stiffness[None] * R[0, 2] * R[0, 2]
        A[0, 1] = shearing_stiffness[None] * R[0, 2] * R[1, 2]
        A[0, 2] = shearing_stiffness[None] * R[0, 2] * R[2, 2]
        A[1, 1] = shearing_stiffness[None] * R[1, 2] * R[1, 2]
        A[1, 2] = shearing_stiffness[None] * R[1, 2] * R[2, 2]
        A[2, 2] = (float(R[2, 2] <= 1) * (-contact_stiffness[None]) * ((1 - R[2, 2]) ** 2)) * R[2, 2]
        A[1, 0] = A[0, 1]
        A[2, 0] = A[0, 2]
        A[2, 1] = A[1, 2]

        A[1, 0] = A[0, 1]
        A[2, 0] = A[0, 2]
        A[2, 1] = A[1, 2]
        garment_mu, garment_la = garment_E[None] / (2 * (1 + garment_nu[None])), garment_E[None] * \
                                 garment_nu[None] / (
                                         (1 + garment_nu[None]) * (1 - 2 * garment_nu[None]))

        fixed_corotated_inplane_J = inplane_F_tmp[p].determinant()
        fixed_corotated_inplane_R = inplane_U[p] @ inplane_V[p].transpose()
        fixed_corotated_P = 2 * garment_mu * (inplane_F_tmp[p] - fixed_corotated_inplane_R) + garment_la * (
                fixed_corotated_inplane_J - 1) * fixed_corotated_inplane_J * inplane_F_tmp[p].transpose().inverse()
        quadrature_P = Q @ ti.Matrix([[fixed_corotated_P[0, 0], fixed_corotated_P[0, 1], 0],
                                      [fixed_corotated_P[1, 0], fixed_corotated_P[1, 1], 0],
                                      [0, 0, 0]]) + Q @ A @ R.inverse().transpose()

        quadrature_P_col3 = ti.Vector([quadrature_P[0, 2], quadrature_P[1, 2], quadrature_P[2, 2]])
        direction = quadrature_d[f, p]
        direction_col3 = ti.Vector([direction[0, 2], direction[1, 2], direction[2, 2]])
        initial_d_inv = initial_quadrature_d_inv[p]

        initial_d_inv_raw1 = ti.Vector([initial_d_inv[0, 0], direction[0, 1], direction[0, 2]])
        initial_d_inv_raw2 = ti.Vector([initial_d_inv[1, 0], direction[1, 1], direction[1, 2]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    offset = ti.Vector([i, j, k])
                    x_gradient_w = ti.Vector(
                        [x_jacobian_w[i, 0] * w[j][1] * w[k][2], w[i][0] * x_jacobian_w[j, 1] * w[k][2],
                         w[i][0] * w[j][1] * x_jacobian_w[k, 2]])
                    v1_weight = v1_w[i][0] * v1_w[j][1] * v1_w[k][2]
                    v2_weight = v2_w[i][0] * v2_w[j][1] * v2_w[k][2]
                    v3_weight = v3_w[i][0] * v3_w[j][1] * v3_w[k][2]
                    weight = w[i][0] * w[j][1] * w[k][2]
                    dpos = (ti.cast(ti.Vector([i, j, k]), real) - fx) * dx
                    ti.atomic_add(grid_v[base + offset], weight * (quadrature_p_mass * temp_v + affine @ dpos))
                    ti.atomic_add(grid_m[base + offset], weight * quadrature_p_mass)
                    ti.atomic_add(grid_v[base + offset],
                                  ((-1) * quadrature_p_vol * quadrature_P_col3 * x_gradient_w.dot(direction_col3)) * dt)
                    ti.atomic_add(grid_v[v1_base + offset], (quadrature_p_vol * v1_weight * (
                            quadrature_P @ (initial_d_inv_raw1 + initial_d_inv_raw2))) * dt)
                    ti.atomic_add(grid_v[v2_base + offset],
                                  ((-1) * quadrature_p_vol * v2_weight * (quadrature_P @ (initial_d_inv_raw1))) * dt)
                    ti.atomic_add(grid_v[v3_base + offset],
                                  ((-1) * quadrature_p_vol * v3_weight * (quadrature_P @ (initial_d_inv_raw2))) * dt)


@ti.kernel
def grid_op(f: ti.i32):
    for i, j, k in grid_m:

        inv_m = 1 / (grid_m[i, j, k] + 1e-9)
        v_out = inv_m * grid_v[i, j, k]

        if i < bound and v_out[0] < 0:
            v_out[0] = 0
            v_out[1] = 0
            v_out[2] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
            v_out[1] = 0
            v_out[2] = 0

        if k < bound and v_out[2] < 0:
            v_out[0] = 0
            v_out[1] = 0
            v_out[2] = 0
        if k > n_grid - bound and v_out[2] > 0:
            v_out[0] = 0
            v_out[1] = 0
            v_out[2] = 0
        if j < bound and v_out[1] < 0:
            v_out[1] = 0
        if j > n_grid - bound and v_out[1] > 0:
            v_out[0] = 0
            v_out[1] = 0
            v_out[2] = 0

        grid_v_out[i, j, k] = v_out


@ti.kernel
def g2p(f: ti.i32):

    for p in range(n_cube_particles):
        base = ti.cast(cube_x[f, p] * inv_dx - 0.5, ti.i32)
        fx = cube_x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_cube_v = ti.Vector([0.0, 0.0, 0.0])
        new_cube_C = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    dpos = ti.cast(ti.Vector([i, j, k]), real) - fx
                    g_v = grid_v_out[base[0] + i, base[1] + j, base[2] + k]
                    weight = w[i][0] * w[j][1] * w[k][2]
                    new_cube_v += weight * g_v
                    new_cube_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        cube_v[f + 1, p] = new_cube_v
        cube_x[f + 1, p] = cube_x[f, p] + dt * cube_v[f + 1, p]
        cube_C[f + 1, p] = new_cube_C



    for p in range(n_operator_particles):
        base = ti.cast(operator_x[f, p] * inv_dx - 0.5, ti.i32)
        fx = operator_x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_operator_v = ti.Vector([0.0, 0.0, 0.0])
        new_operator_C = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    dpos = ti.cast(ti.Vector([i, j, k]), real) - fx
                    g_v = grid_v_out[base[0] + i, base[1] + j, base[2] + k]
                    weight = w[i][0] * w[j][1] * w[k][2]
                    new_operator_v += weight * g_v
                    new_operator_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        operator_v[f + 1, p] = new_operator_v
        operator_x[f + 1, p] = operator_x[f, p] + dt * operator_v[f + 1, p]
        operator_C[f + 1, p] = new_operator_C

    for p in range(n_vertex_particles):
        base = ti.cast(vertex_x[f, p] * inv_dx - 0.5, ti.i32)
        fx = vertex_x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_vertex_v = ti.Vector([0.0, 0.0, 0.0])
        new_vertex_C = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    dpos = ti.cast(ti.Vector([i, j, k]), real) - fx
                    g_v = grid_v_out[base[0] + i, base[1] + j, base[2] + k]
                    weight = w[i][0] * w[j][1] * w[k][2]
                    new_vertex_v += weight * g_v
                    new_vertex_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        vertex_v[f + 1, p] = new_vertex_v
        vertex_x[f + 1, p] = vertex_x[f, p] + dt * vertex_v[f + 1, p]
        vertex_C[f + 1, p] = new_vertex_C

    for p in range(n_quadrature_particles):
        base = ti.cast(quadrature_x[f, p] * inv_dx - 0.5, ti.i32)
        fx = quadrature_x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_quadrature_C = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    dpos = ti.cast(ti.Vector([i, j, k]), real) - fx
                    g_v = grid_v_out[base[0] + i, base[1] + j, base[2] + k]
                    weight = w[i][0] * w[j][1] * w[k][2]
                    new_quadrature_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        C_skew_part = 0.5 * (new_quadrature_C - new_quadrature_C.transpose())
        C_sym_part = 0.5 * (new_quadrature_C + new_quadrature_C.transpose())
        quadrature_C[f + 1, p] = C_skew_part + (1 - 0.95) * C_sym_part


@ti.kernel
def update_particle_state(f: ti.i32):
    for p in range(n_quadrature_particles):
        v1 = lagrangian_mesh_element[p][0]
        v2 = lagrangian_mesh_element[p][1]
        v3 = lagrangian_mesh_element[p][2]
        quadrature_v[f + 1, p] = (1 / 3) * (vertex_v[f + 1, v1] + vertex_v[f + 1, v2] + vertex_v[f + 1, v3])
        quadrature_x[f + 1, p] = (1 / 3) * (vertex_x[f + 1, v1] + vertex_x[f + 1, v2] + vertex_x[f + 1, v3])

        new_quadrature_d_col1 = vertex_x[f + 1, v2] - vertex_x[f + 1, v1]
        new_quadrature_d_col2 = vertex_x[f + 1, v3] - vertex_x[f + 1, v1]
        new_quadrature_d_col3 = ti.Vector(
            [quadrature_d[f, p][0, 2], quadrature_d[f, p][1, 2], quadrature_d[f, p][2, 2]])
        new_quadrature_d_col3 += (dt * quadrature_C[f + 1, p] + 0.5 * dt * dt * quadrature_C[f + 1, p] @ quadrature_C[
            f + 1, p]) @ new_quadrature_d_col3

        new_quadrature_d = ti.Matrix.cols([new_quadrature_d_col1, new_quadrature_d_col2, new_quadrature_d_col3])

        initial_d_inv = initial_quadrature_d_inv[p]

        # return mapping
        new_F = new_quadrature_d @ initial_d_inv
        Q, R = QR_decomposition_3d(new_F)
        cf = 0.1
        if (shearing_stiffness[None] == 0):
            R[0, 2] = 1e-7
            R[1, 2] = 1e-7
            R[2, 2] = min(R[2, 2], 1)
        elif (R[2, 2] > 1):
            R[0, 2] = 1e-7
            R[1, 2] = 1e-7
            R[2, 2] = 1
        elif (R[2, 2] < 0):
            R[0, 2] = 1e-7
            R[1, 2] = 1e-7
            R[2, 2] = max(R[2, 2], -1)
        elif (R[2, 2] <= 1):
            shear = (R[0, 2] ** 2) + (R[1, 2] ** 2)
            shearing_stiffness_over_k = shearing_stiffness[None] / contact_stiffness[None]
            if ((shearing_stiffness_over_k ** 2) * shear - ((cf * (R[2, 2] - 1) ** 2) ** 2) > 0):
                return_mapping_scale = cf * ((R[2, 2] - 1) ** 2) / (shearing_stiffness_over_k * ti.sqrt(shear))
                R[0, 2] = R[0, 2] * return_mapping_scale
                R[1, 2] = R[1, 2] * return_mapping_scale
        quadrature_d[f + 1, p] = new_F @ initial_d_inv.inverse()
        quadrature_F[f + 1, p] = new_F




def initialize_objects():
    grasp_point_model = operator_model_without_position + grasp_point_pos[0][0]
    grasp_tracking_seq.from_numpy(grasp_point_pos)
    for i in range(grasp_point_num - 1):
        grasp_point_model = np.concatenate(
            (grasp_point_model, (operator_model_without_position + grasp_point_pos[0][i + 1])), axis=0)
    initial_operator_x.from_numpy(grasp_point_model)
    initial_vertex_x.from_numpy(soft_body_model_vertices)
    lagrangian_mesh_element.from_numpy(soft_body_model_triangles)
    initial_cube_x.from_numpy(cube_model)
    initialize_ti_field()


@ti.kernel
def initialize_ti_field():
    for i in range(n_cube_particles):
        cube_x[0, i] = initial_cube_x[i]
        cube_v[0, i] = [0, 0, 0]
        cube_C[0, i] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        cube_F[0, i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for i in range(n_operator_particles):
        operator_x[0, i] = initial_operator_x[i]
        operator_v[0, i] = [0, 0, 0]
    for i in range(n_vertex_particles):
        vertex_x[0, i] = initial_vertex_x[i]
        vertex_v[0, i] = [0, 0, 0]
        vertex_C[0, i] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for i in range(n_quadrature_particles):
        index0 = lagrangian_mesh_element[i][0]
        index1 = lagrangian_mesh_element[i][1]
        index2 = lagrangian_mesh_element[i][2]
        quadrature_x[0, i] = ((initial_vertex_x[index0] + initial_vertex_x[index1] + initial_vertex_x[index2]) / 3)
        quadrature_v[0, i] = [0, 0, 0]
        quadrature_C[0, i] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        quadrature_F[0, i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        quadrature_d_col1 = (initial_vertex_x[index1] - initial_vertex_x[index0])
        quadrature_d_col2 = (initial_vertex_x[index2] - initial_vertex_x[index0])
        normal_quadrature_d_col1 = quadrature_d_col1 / quadrature_d_col1.norm(1e-6)
        normal_quadrature_d_col2 = quadrature_d_col2 / quadrature_d_col2.norm(1e-6)
        quadrature_d_col3 = ti.Matrix.cross(normal_quadrature_d_col1, normal_quadrature_d_col2)
        normal_quadrature_d_col3 = quadrature_d_col3 / quadrature_d_col3.norm(1e-6)
        initial_d = ti.Matrix.cols([quadrature_d_col1, quadrature_d_col2, normal_quadrature_d_col3])
        quadrature_d[0, i] = initial_d
        initial_quadrature_d_inv[i] = initial_d.inverse()


@ti.kernel
def compute_loss(f: ti.i32):

    for p in range(n_vertex_particles):
        for q in range(n_target_particles):
            dist = (vertex_x[f - 1, p] - target_pos[q]) ** 2
            dist_norm = (dist[0] + dist[1] + dist[2]) / 3
            distances[p, q] = dist_norm
    for q in range(n_target_particles):
        min_distance = distances[0, q]
        for p in range(1, n_vertex_particles):
            if distances[p, q] < min_distance:
                min_distance = distances[p, q]
        loss[None] += min_distance / n_target_particles


@ti.ad.grad_replaced
def forward(s, grad_update):
    clear_grid()
    compute_inplane_F_tmp(s)
    inplane_svd()
    p2g(s, grad_update)
    grid_op(s)
    g2p(s)
    update_particle_state(s)



@ti.ad.grad_for(forward)
def backward(s, grad_update):
    clear_grid()
    clear_inplane_SVD_grad()
    compute_inplane_F_tmp(s)
    inplane_svd()
    p2g(s, grad_update)
    grid_op(s)
    g2p(s)
    update_particle_state(s)
    update_particle_state.grad(s)

    g2p.grad(s)
    grid_op.grad(s)
    p2g.grad(s, grad_update)
    inplane_svd_grad()
    compute_inplane_F_tmp.grad(s)

def compute_grad():
    losses = []
    garment_E_grad, garment_nu_grad, contact_stiffness_grad, shearing_stiffness_grad = [], [], [], []
    for grad_updates in range(n_grad_updates):
        clear_distances()
        target_pos.from_numpy(target_pcd_array[grad_updates])
        print("Grad_update:", grad_updates, "/", n_grad_updates)
        timeC = time.time()
        with ti.ad.Tape(loss=loss):
            loss[None] = 0
            for f in range(max_steps - 1):
                forward(f, grad_updates)
            compute_loss(f)
        timeD = time.time()
        print("Loss:", loss[None])
        print("Grads:", garment_E.grad[None], garment_nu.grad[None], contact_stiffness.grad[None], shearing_stiffness.grad[None])
        print("time_cost:", timeD - timeC)
        garment_E_grad.append(garment_E.grad[None])
        garment_nu_grad.append(garment_nu.grad[None])
        contact_stiffness_grad.append(contact_stiffness.grad[None])
        shearing_stiffness_grad.append(shearing_stiffness.grad[None])
        losses.append(loss[None])



        garment_vertices = vertex_x.to_numpy()[max_steps - 1]
        garment_faces = np.asarray(soft_body_mesh.triangles)
        operator_pos = operator_x.to_numpy()[max_steps - 1]
        cube_pos = cube_x.to_numpy()[max_steps - 1]

        initial_operator_x.from_numpy(operator_pos)
        initial_vertex_x.from_numpy(garment_vertices)
        lagrangian_mesh_element.from_numpy(garment_faces)
        initial_cube_x.from_numpy(cube_pos)
        initialize_ti_field()
    grad_result_list = []
    print(losses)
    loss_mean = sum(losses) / n_grad_updates
    print(loss_mean)
    grad_result_list.append(np.mean(np.nan_to_num(garment_E_grad)))
    grad_result_list.append(np.mean(np.nan_to_num(garment_nu_grad)))
    grad_result_list.append(np.mean(np.nan_to_num(contact_stiffness_grad)))
    grad_result_list.append(np.mean(np.nan_to_num(shearing_stiffness_grad)))

    return loss_mean, grad_result_list


def magnitude_level(a):
    log = math.log10(abs(a + 1e-19))
    level = math.floor(log)
    return level


def single_parameter_fitting(parameter, parameter_grad, grad_shock, lower_bound, upper_bound, lr, epochs, epoch_index):
    period = epochs - epoch_index
    if abs(parameter_grad) > 0:
        delta = lr * parameter_grad
        if parameter_grad < 0:
            new_magnitude = magnitude_level((upper_bound - parameter) / period)
        if parameter_grad > 0:
            new_magnitude = magnitude_level((lower_bound - parameter) / period)
        if epoch_index < 1:
            lr *= pow(10, new_magnitude - magnitude_level(delta))
            delta = lr * parameter_grad
        parameter_update = parameter - delta
        if lower_bound < parameter_update < upper_bound:
            parameter = parameter_update
            lr *= pow(10, new_magnitude - magnitude_level(delta))
        else:
            if parameter_update < lower_bound:
                new_magnitude = magnitude_level((upper_bound - parameter) / period)
            if parameter_update > upper_bound:
                new_magnitude = magnitude_level((lower_bound - parameter) / period)
            lr *= pow(10, new_magnitude - magnitude_level(delta))
            parameter_update = parameter - lr * parameter_grad
            while parameter_update > upper_bound or parameter_update < lower_bound:
                lr *= 0.5
                parameter_update = parameter - lr * parameter_grad
            parameter = parameter_update

    return parameter, lr





def parameters_learning(elasticity_parameter_learning: bool,
                        contact_parameter_learning: bool, epoch: ti.i32):
    print()
    learning_rate0[None] = 1
    learning_rate1[None] = 1
    learning_rate2[None] = 1
    learning_rate3[None] = 1

    garment_E_grad = 0.0
    garment_nu_grad = 0.0
    contact_stiffness_grad = 0.0
    shearing_stiffness_grad = 0.0

    min_loss = 1e9

    for i in range(epoch):
        print()
        print("epoch: ", i)

        G = garment_E[None] / (2 + 2 * garment_nu[None])
        print("E =", garment_E[None], "\nNu =", garment_nu[None], "\nG =", G)
        miu, lam = garment_E[None] / (2 * (1 + garment_nu[None])), garment_E[None] * garment_nu[
            None] / ((1 + garment_nu[None]) * (1 - 2 * garment_nu[None]))
        print("miu =", miu, "\nlambda =", lam)
        print('contact_stiffness =', contact_stiffness[None], '\nshearing_stiffness =', shearing_stiffness[None])


        print()
        initialize_objects()
        timeA = time.time()
        l, grads = compute_grad()
        timeB = time.time()
        print('>>> loss =', l)
        print('>>> cost_time =', timeB - timeA)
        print()

        grad_shock0 = 0
        grad_shock1 = 0
        grad_shock2 = 0
        grad_shock3 = 0
        if (garment_E_grad * grads[0] < 0):
            grad_shock0 = 1
        if (garment_nu_grad * grads[1] < 0):
            grad_shock1 = 1
        if (contact_stiffness_grad * grads[2] < 0):
            grad_shock2 = 1
        if (shearing_stiffness_grad * grads[3] < 0):
            grad_shock3 = 1
        if (min_loss >= l or i == 1):
            min_loss = l
            best_E = garment_E[None]
            best_nu = garment_nu[None]
            best_contact_stiffness = contact_stiffness[None]
            best_shearing_stiffness = shearing_stiffness[None]


        if ((abs(garment_E.grad[None]) >= 0) == False):
            print("Gradients disappear!")
            break

        if (elasticity_parameter_learning == True):
            garment_E[None], learning_rate0[None] = single_parameter_fitting(garment_E[None],
                                                                             grads[0], grad_shock0,
                                                                             E_lower_bound, E_upper_bound,
                                                                             learning_rate0[None], epoch, i)
            garment_nu[None], learning_rate1[None] = single_parameter_fitting(garment_nu[None],
                                                                              grads[1], grad_shock1,
                                                                              nu_lower_bound, nu_upper_bound,
                                                                              learning_rate1[None], epoch, i)
            print('garment_E =', garment_E[None])
            print('garment_nu =', garment_nu[None])


        if (contact_parameter_learning == True):
            contact_stiffness[None], learning_rate2[None] = single_parameter_fitting(contact_stiffness[None],
                                                                                     grads[2],
                                                                                     grad_shock2,
                                                                                     contact_stiffness_lower_bound,
                                                                                     contact_stiffness_upper_bound,
                                                                                     learning_rate2[None], epoch, i)
            shearing_stiffness[None], learning_rate3[None] = single_parameter_fitting(shearing_stiffness[None],
                                                                                      grads[3],
                                                                                      grad_shock3,
                                                                                      shearing_stiffness_lower_bound,
                                                                                      shearing_stiffness_upper_bound,
                                                                                      learning_rate3[None], epoch, i)


            print('contact_stiffness =', contact_stiffness[None])
            print('shearing_stiffness =', shearing_stiffness[None])




        print()

        if ((abs(loss[None]) > 0) == False):
            print("loss is NaN!")
            print("ERROR!")
            print("ERROR!")
            print("ERROR!")

    garment_E[None] = best_E
    garment_nu[None] = best_nu
    contact_stiffness[None] = best_contact_stiffness
    shearing_stiffness[None] = best_shearing_stiffness
    loss[None] = min_loss

    print("Result:")
    miu, lam = garment_E[None] / (2 * (1 + garment_nu[None])), garment_E[None] * garment_nu[None] / (
            (1 + garment_nu[None]) * (1 - 2 * garment_nu[None]))
    print("  min loss is", min_loss)
    print("  prediction E is", garment_E[None])
    print("  prediction nu is", garment_nu[None])
    print("  prediction miu is", miu)
    print("  prediction lambda is", lam)
    print("  prediction contact_stiffness is", contact_stiffness[None])
    print("  prediction shearing_stiffness is", shearing_stiffness[None])
    print()


def initialize_constitutive_param():
    garment_E[None] = initial_garment_E
    garment_nu[None] = initial_garment_nu
    contact_stiffness[None] = initial_contact_stiffness
    shearing_stiffness[None] = initial_shearing_stiffness



print()

initial_garment_E = 100
initial_garment_nu = 0.05
initial_contact_stiffness = 100
initial_shearing_stiffness = 100
initialize_constitutive_param()

print("Initializing objects...")
initialize_objects()
print("Initialized objects")

time0 = time.time()
parameters_learning(True, True, 35)
time1 = time.time()

garment_nu[None] = initial_garment_nu
contact_stiffness[None] = initial_contact_stiffness
shearing_stiffness[None] = initial_shearing_stiffness

result_file = open(result_file_path)
result = json.load(result_file)
result['E'] = garment_E[None]
result['nu'] = garment_nu[None]
result['contact_stiffness'] = contact_stiffness[None]
result['shearing_stiffness'] = shearing_stiffness[None]
with open(result_file_path, 'w') as f:
    json.dump(result, f, indent=1)

