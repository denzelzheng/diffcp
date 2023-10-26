from src.taichi_variable_definition_without_grad import *

@ti.kernel
def clear_grid():
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0, 0, 0]
        grid_m[i, j, k] = 0

def zero_vec():
    return [0.0, 0.0, 0.0]

def zero_matrix():
    return [zero_vec(), zero_vec(), zero_vec()]

@ti.kernel
def compute_inplane_F_tmp(f: ti.i32):
    for p in range(0, n_quadrature_particles):
        Q, R = QR_decomposition_3d(quadrature_F[p])
        inplane_F_tmp[p] = ti.Matrix([[R[0, 0], R[0, 1]], [R[1, 0], R[1, 1]]])

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
def p2g(f: ti.i32):
    for p in range(0, n_cube_particles):
        base = ti.cast(cube_x[p] * inv_dx - 0.5, ti.i32)
        fx = cube_x[p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        affine = cube_p_mass * cube_C[p]
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
        base = ti.cast(operator_x[p] * inv_dx - 0.5, ti.i32)
        fx = operator_x[p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        affine = operator_p_mass * operator_C[p]

        t = f * dt
        frame = int(t / operator_update_interval)
        grasp_id = int(p / grasper_group_size)
        operator_temp_v = ti.Vector([0.0, 0.0, 0.0])
        if frame + 1 < n_actions:
            operator_temp_v = (grasp_tracking_seq[frame + 1, grasp_id] - grasp_tracking_seq[frame, grasp_id]) / operator_update_interval
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
        base = ti.cast(vertex_x[p] * inv_dx - 0.5, ti.i32)
        fx = vertex_x[p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        affine = vertex_p_mass * vertex_C[p]
        temp_v = vertex_v[p]
        temp_v = vertex_v[p] + [0, dt * scene_scaling * gravity, 0]
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
        base = ti.cast(quadrature_x[p] * inv_dx - 0.5, ti.i32)
        fx = quadrature_x[p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        x_jacobian_w = ti.Matrix.rows([fx - 1.5, 2 * (1.0 - fx), fx - 0.5]) * inv_dx
        affine = quadrature_C[p] * quadrature_p_mass
        temp_v = quadrature_v[p]

        v1 = lagrangian_mesh_element[p][0]
        v2 = lagrangian_mesh_element[p][1]
        v3 = lagrangian_mesh_element[p][2]
        v1_base = ti.cast(vertex_x[v1] * inv_dx - 0.5, ti.i32)
        v1_fx = vertex_x[v1] * inv_dx - ti.cast(v1_base, ti.i32)
        v1_w = [0.5 * (1.5 - v1_fx) ** 2, 0.75 - (v1_fx - 1) ** 2, 0.5 * (v1_fx - 0.5) ** 2]
        v2_base = ti.cast(vertex_x[v2] * inv_dx - 0.5, ti.i32)
        v2_fx = vertex_x[v2] * inv_dx - ti.cast(v2_base, ti.i32)
        v2_w = [0.5 * (1.5 - v2_fx) ** 2, 0.75 - (v2_fx - 1) ** 2, 0.5 * (v2_fx - 0.5) ** 2]
        v3_base = ti.cast(vertex_x[v3] * inv_dx - 0.5, ti.i32)
        v3_fx = vertex_x[v3] * inv_dx - ti.cast(v3_base, ti.i32)
        v3_w = [0.5 * (1.5 - v3_fx) ** 2, 0.75 - (v3_fx - 1) ** 2, 0.5 * (v3_fx - 0.5) ** 2]

        Q, R = QR_decomposition_3d(quadrature_F[p])
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
        direction = quadrature_d[p]
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
        base = ti.cast(cube_x[p] * inv_dx - 0.5, ti.i32)
        fx = cube_x[p] * inv_dx - ti.cast(base, real)
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

        cube_v[p] = new_cube_v
        cube_x[p] = cube_x[p] + dt * cube_v[p]
        cube_C[p] = new_cube_C

    for p in range(n_operator_particles):
        base = ti.cast(operator_x[p] * inv_dx - 0.5, ti.i32)
        fx = operator_x[p] * inv_dx - ti.cast(base, real)
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

        operator_v[p] = new_operator_v
        operator_x[p] = operator_x[p] + dt * operator_v[p]
        operator_C[p] = new_operator_C

    for p in range(n_vertex_particles):
        base = ti.cast(vertex_x[p] * inv_dx - 0.5, ti.i32)
        fx = vertex_x[p] * inv_dx - ti.cast(base, real)
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

        vertex_v[p] = new_vertex_v
        vertex_x[p] = vertex_x[p] + dt * vertex_v[p]
        vertex_C[p] = new_vertex_C

    for p in range(n_quadrature_particles):
        base = ti.cast(quadrature_x[p] * inv_dx - 0.5, ti.i32)
        fx = quadrature_x[p] * inv_dx - ti.cast(base, real)
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
        quadrature_C[p] = C_skew_part + (1 - 0.95) * C_sym_part


@ti.kernel
def update_particle_state(f: ti.i32):
    for p in range(n_quadrature_particles):
        v1 = lagrangian_mesh_element[p][0]
        v2 = lagrangian_mesh_element[p][1]
        v3 = lagrangian_mesh_element[p][2]
        quadrature_v[p] = (1 / 3) * (vertex_v[v1] + vertex_v[v2] + vertex_v[v3])
        quadrature_x[p] = (1 / 3) * (vertex_x[v1] + vertex_x[v2] + vertex_x[v3])

        new_quadrature_d_col1 = vertex_x[v2] - vertex_x[v1]
        new_quadrature_d_col2 = vertex_x[v3] - vertex_x[v1]

        new_quadrature_d_col3 = ti.Vector(
            [quadrature_d[p][0, 2], quadrature_d[p][1, 2], quadrature_d[p][2, 2]])
        new_quadrature_d_col3 += (dt * quadrature_C[p] + 0.5 * dt * dt * quadrature_C[p] @ quadrature_C[p]) @ new_quadrature_d_col3

        # normal_quadrature_d_col1 = new_quadrature_d_col1 / new_quadrature_d_col1.norm(1e-6)
        # normal_quadrature_d_col2 = new_quadrature_d_col2 / new_quadrature_d_col2.norm(1e-6)
        # quadrature_d_col3 = ti.Matrix.cross(normal_quadrature_d_col1, normal_quadrature_d_col2)
        # new_quadrature_d_col3 = quadrature_d_col3 / quadrature_d_col3.norm(1e-6)

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
        quadrature_d[p] = new_F @ initial_d_inv.inverse()
        quadrature_F[p] = new_F

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
        cube_x[i] = initial_cube_x[i]
        cube_v[i] = [0, 0, 0]
        cube_C[i] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for i in range(n_operator_particles):
        operator_x[i] = initial_operator_x[i]
        operator_v[i] = [0, 0, 0]
        operator_C[i] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for i in range(n_vertex_particles):
        vertex_x[i] = initial_vertex_x[i]
        vertex_v[i] = [0, 0, 0]
        vertex_C[i] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for i in range(n_quadrature_particles):
        index0 = lagrangian_mesh_element[i][0]
        index1 = lagrangian_mesh_element[i][1]
        index2 = lagrangian_mesh_element[i][2]
        quadrature_x[i] = ((initial_vertex_x[index0] + initial_vertex_x[index1] + initial_vertex_x[index2]) / 3)
        quadrature_v[i] = [0, 0, 0]
        quadrature_C[i] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        quadrature_F[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        quadrature_d_col1 = (initial_vertex_x[index1] - initial_vertex_x[index0])
        quadrature_d_col2 = (initial_vertex_x[index2] - initial_vertex_x[index0])
        normal_quadrature_d_col1 = quadrature_d_col1 / quadrature_d_col1.norm(1e-6)
        normal_quadrature_d_col2 = quadrature_d_col2 / quadrature_d_col2.norm(1e-6)
        quadrature_d_col3 = ti.Matrix.cross(normal_quadrature_d_col1, normal_quadrature_d_col2)
        normal_quadrature_d_col3 = quadrature_d_col3 / quadrature_d_col3.norm(1e-6)
        initial_d = ti.Matrix.cols([quadrature_d_col1, quadrature_d_col2, normal_quadrature_d_col3])
        quadrature_d[i] = initial_d
        initial_quadrature_d_inv[i] = initial_d.inverse()

def forward(s):
    clear_grid()
    compute_inplane_F_tmp(s)
    inplane_svd()
    p2g(s)
    grid_op(s)
    g2p(s)
    update_particle_state(s)

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
    tmp_dir = os.path.abspath(os.path.dirname(__file__)) + '/../tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    initial_vertices_file = tmp_dir + "/initial_vertices.npy"
    np.save(initial_vertices_file, soft_body_model_vertices)

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
            if (count == int((animation_time / IFS) * i / n_pose_estimation_output - 2)):
                garment_mesh.vertices = o3d.utility.Vector3dVector(vertex_x.to_numpy())
                garment_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(soft_body_mesh.triangles))
                garment_mesh.compute_vertex_normals()
                garment_mesh.paint_uniform_color([x / 255 for x in [30, 141, 255]])
                save_mesh_path = current_dir + result_dir + "/pose_estimation_output/" + str(i) + ".ply"
                o3d.io.write_triangle_mesh(save_mesh_path, garment_mesh)