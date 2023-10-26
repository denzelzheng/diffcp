import taichi as ti
import numpy as np
import random
import taichi_glsl as ts
import math
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
arguments_file = open(current_dir + '/simulation_arguments.json')
arguments = json.load(arguments_file)
operator_parameters = arguments['operator_parameters']
operator_update_interval = arguments['operator_update_interval']
dt = arguments['dt']
n_grid = arguments['n_grid']
scene_scaling = arguments['scene_scaling']
initial_frame = arguments['initial_frame']
end_frame = arguments['end_frame']
operator_edge_length = operator_parameters['operator_edge_length']
n_operator_particles = operator_parameters['n_operator_particles']
garment_parameters = arguments['garment_parameters']
upper_bounds = garment_parameters['constitutive_parameter_upper_bounds']
lower_bounds = garment_parameters['constitutive_parameter_lower_bounds']
data_dir = arguments['data_dir']
result_dir = arguments['result_dir']
n_pose_estimation_output = arguments['n_pose_estimation_output']
number_of_triangles = garment_parameters['number_of_triangles']
operator_mass = operator_parameters['operator_mass']
rigid_cube_parameters = arguments['rigid_cube_parameters']
cube_mass = rigid_cube_parameters['cube_mass']
rigid_cube_side_length = rigid_cube_parameters['rigid_cube_side_length']
grad_update_interval = arguments['grad_update_interval']
E_lower_bound = lower_bounds['E']
nu_lower_bound = lower_bounds['nu']
contact_stiffness_lower_bound = lower_bounds['contact_stiffness']
shearing_stiffness_lower_bound = lower_bounds['shearing_stiffness']
n_rigid_cube_particles = rigid_cube_parameters['n_rigid_cube_particles']

E_upper_bound = upper_bounds['E']
nu_upper_bound = upper_bounds['nu']
contact_stiffness_upper_bound = upper_bounds['contact_stiffness']
shearing_stiffness_upper_bound = upper_bounds['shearing_stiffness']

initial_garment_parameter = garment_parameters['initial_constitutive_parameters']
initial_garment_E = initial_garment_parameter['E']
initial_garment_nu = initial_garment_parameter['nu']
initial_contact_stiffness = initial_garment_parameter['contact_stiffness']
initial_shearing_stiffness = initial_garment_parameter['shearing_stiffness']

garment_mass = garment_parameters['garment_mass']
garment_mass = garment_mass * (scene_scaling ** 3)

data_dir = current_dir + data_dir
result_file_path = current_dir + result_dir + '/fit_parameter.json'
cloth_file = data_dir + '/cloth_mesh.ply'
mark_file = data_dir + '/rigid_cube_top_face_center_position.npy'
world_target_path = data_dir + '/target_pcd_sample.npy'
target_pcd_array = np.load(world_target_path)
mark_pos = np.load(mark_file)
grasp_seq = np.load(data_dir + "/grasp_traj_array.npy")

dx, inv_dx = 1 / n_grid, float(n_grid)
initial_step = 1
n_sample_target_points = target_pcd_array.shape[1]
n_grad_updates = target_pcd_array.shape[0]
n_target_particles = target_pcd_array.shape[1]

random_points = np.random.rand(n_rigid_cube_particles, 3) * rigid_cube_side_length - rigid_cube_side_length / 2
random_points[:, 0] += mark_pos[0]
random_points[:, 1] += mark_pos[1] - rigid_cube_side_length/2
random_points[:, 2] += mark_pos[2]
cube_model = random_points

frame_range = grasp_seq.shape[0]
print("There are", frame_range, "frames in data")
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

soft_body_model_vertices = np.asarray(soft_body_mesh.vertices) * scene_scaling
soft_body_mesh.vertices = o3d.utility.Vector3dVector(soft_body_model_vertices)
soft_body_model_triangles = np.asarray(soft_body_mesh.triangles)
grasp_seq = grasp_seq * scene_scaling
cube_model = cube_model * scene_scaling
target_pcd_array = target_pcd_array * scene_scaling
garment_surface_area = soft_body_mesh.get_surface_area()
n_vertex_particles = soft_body_model_vertices.shape[0]
n_quadrature_particles = soft_body_model_triangles.shape[0]
n_operator_particles = int(n_operator_particles / grasp_point_num) * grasp_point_num
n_cube_particles = cube_model.shape[0]

garment_rho = garment_mass / (garment_surface_area * dx)
operator_edge_length *= scene_scaling
operator_volume = operator_edge_length ** 3
rigid_cube_side_length *= scene_scaling
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
soft_body_mesh.vertices = o3d.utility.Vector3dVector(soft_body_model_vertices)

particle_position_series = list()
operator_particle_position_series = list()
target_particle_position_series = list()
operator_target_particle_position_series = list()
min_loss, best_E, best_nu = 0, 0, 0
max_steps = int(grad_update_interval / dt) + 1

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