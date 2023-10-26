from src.simulation import *

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

