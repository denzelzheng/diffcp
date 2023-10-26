from src.simulation_without_grad import *

@ti.kernel
def reset_quadrature_particles():
    for i in range(n_quadrature_particles):
        index0 = lagrangian_mesh_element[i][0]
        index1 = lagrangian_mesh_element[i][1]
        index2 = lagrangian_mesh_element[i][2]
        quadrature_x[i] = ((vertex_x[index0] + vertex_x[index1] + vertex_x[index2]) / 3)
        quadrature_d_col1 = (vertex_x[index1] - vertex_x[index0])
        quadrature_d_col2 = (vertex_x[index2] - vertex_x[index0])
        normal_quadrature_d_col1 = quadrature_d_col1 / quadrature_d_col1.norm(1e-6)
        normal_quadrature_d_col2 = quadrature_d_col2 / quadrature_d_col2.norm(1e-6)
        quadrature_d_col3 = ti.Matrix.cross(normal_quadrature_d_col1, normal_quadrature_d_col2)
        normal_quadrature_d_col3 = quadrature_d_col3 / quadrature_d_col3.norm(1e-6)
        tmp_d = ti.Matrix.cols([quadrature_d_col1, quadrature_d_col2, normal_quadrature_d_col3])
        quadrature_d[i] = tmp_d
        initial_quadrature_d_inv[i] = tmp_d.inverse()

def animation(animation_time: ti.f32):
    print("Simulating the process for visualization...")
    G = garment_E[None] / (2 + 2 * garment_nu[None])
    print("E =", garment_E[None], "\nNu =", garment_nu[None], "\nG =", G)
    miu, lam = garment_E[None] / (2 * (1 + garment_nu[None])), garment_E[None] * garment_nu[None] / (
            (1 + garment_nu[None]) * (1 - 2 * garment_nu[None]))
    print("miu =", miu, "\nlambda =", lam)
    print('contact_stiffness =', contact_stiffness[None], '\nshearing_stiffness =', shearing_stiffness[None])

    IFS = 0.001
    initial_mesh = o3d.geometry.TriangleMesh()
    initial_mesh.vertices = o3d.utility.Vector3dVector(soft_body_model_vertices)
    initial_mesh.triangles = o3d.utility.Vector3iVector(soft_body_model_triangles)

    for count in range(int(animation_time / IFS)):
        for f in range(count * int(IFS / dt), (count + 1) * int(IFS / dt)):
            forward(f)
        reset_quadrature_particles()
        
        garment_mesh = o3d.geometry.TriangleMesh()
        for i in range(n_pose_estimation_output + 1):
            if (count == int((animation_time / IFS)  * i / n_pose_estimation_output - 2) ):
                garment_mesh.vertices = o3d.utility.Vector3dVector(vertex_x.to_numpy())
                garment_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(soft_body_mesh.triangles))
                garment_mesh.compute_vertex_normals()
                garment_mesh.paint_uniform_color([x / 255 for x in [30, 141, 255]])
                save_mesh_path = current_dir + result_dir + "/pose_estimation_output/" + str(i) + ".ply"
                o3d.io.write_triangle_mesh(save_mesh_path, garment_mesh)

fit_parameter = json.load(open(current_dir + result_dir + "/fitting_result.json"))
garment_E[None] = fit_parameter['E']
garment_nu[None] = fit_parameter['nu']
contact_stiffness[None] = fit_parameter['contact_stiffness']
shearing_stiffness[None] = fit_parameter['shearing_stiffness']
print("Initializing objects...")
initialize_objects()
print("Initialized objects")
animation(operator_update_interval * (frame_range - 1))
