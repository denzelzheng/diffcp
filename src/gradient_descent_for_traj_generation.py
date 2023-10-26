from src.trajectory_generation_functions_and_variables import *
from src.parameter_fitting_functions import *

n_actions = 31
lr_setting = 1e-13
learning_rate = [lr_setting] * n_actions

def compute_grad():
    cd = 0
    for grad_updates in range(1):
        clear_distances()
        target_pos.from_numpy(target)
        timeC = time.time()
        with ti.ad.Tape(loss=loss):
            loss[None] = 0
            for f in range(max_steps - 1):
                forward(f, grad_updates)
            compute_loss(f)
        timeD = time.time()
        print("Loss:", loss[None])
        cd = compute_chamfer_distance(target, vertex_x.to_numpy())
    grad_result_list = []
    for action_index in range(n_actions):
        grad_result_list.append(np.nan_to_num(grasp_tracking_seq.grad[action_index]))
    grad_result_list = np.array(grad_result_list)
    return loss, grad_result_list, cd

def traj_optimization(epoch: ti.i32):
    print()
    chamfer_list = []
    loss_list = []
    for i in range(n_actions):
        learning_rate[i] = lr_setting
    for i in range(epoch):
        print("epoch: ", i)
        initialize_objects()
        l, grads, chamfer = compute_grad()
        loss_list.append(l[None])
        chamfer_list.append(chamfer)
        print('chamfer distance =', chamfer)
        global grasp_point_pos
        for j in range(n_actions):
            for k in range(3):
                grasp_tracking_seq[j][k] -= learning_rate[j] * grads[j][k]
        grasp_point_pos = grasp_tracking_seq.to_numpy()
        new_traj = grasp_point_pos
    return np.array(loss_list), np.array(chamfer_list), new_traj

fit_parameter = json.load(open(current_dir + result_dir + "/fitting_result.json"))
garment_E[None] = fit_parameter['E']
garment_nu[None] = fit_parameter['nu']
contact_stiffness[None] = fit_parameter['contact_stiffness']
shearing_stiffness[None] = fit_parameter['shearing_stiffness']
print("Initializing objects...")
initialize_objects()
print("Initialized objects")

loss_arr = []
chamfer_arr = []
trajectories = []
for i in range(5):
    lr_setting *= 0.1
    losses, chamfer_distances, new_trajectory = parameters_learning(500)
    loss_arr.append(losses)
    chamfer_arr.append(chamfer_distances)
    trajectories.append(new_trajectory)
loss_arr = np.array(loss_arr)
chamfer_arr = np.array(chamfer_arr)
np.save(result_dir + "/GD_chamfer_distances.npy", chamfer_arr)
np.save(result_dir + "/GD_loss.npy", loss_arr)
np.save(result_dir + "/GD_trajectories.npy", trajectories)