from src.simulation_without_grad import *
from src.trajectory_generation_functions_and_variables import *
import gym
from stable_baselines3 import SAC
from gym import spaces
from stable_baselines3.common.callbacks import BaseCallback

n_step_per_ep = 31
n_ep = 500
traj_length = 31
total_time = 0.3 / n_step_per_ep
grasp_seq = np.zeros((traj_length, 1, 3))

class CustomEnv(gym.Env):
    def __init__(self, n):
        self.n = n
        low = np.array([0.3, 0, -0.5])
        high = np.array([1, 0.45, 0.5])
        action_range = 2.1 / n
        self.observation_space = spaces.Box(low=low, high=high, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(low=-action_range, high=action_range, shape=(3,), dtype=np.float32)
        self.reward_arr = []
        self.step_counter = 0
        self.ep_counter = -1
        self.max_steps = n
        self.step_rewards = {}
        self.ep_loss = []
        self.reward_for_ep = 0
        self.cd = 0
        self.traj = []
        self.eps_traj = []

    def reset(self):
        self.state = np.array([0.8, 0, -0.3])
        self.traj = [self.state]
        self.step_counter = 0
        self.ep_counter += 1
        self.reward_for_ep = 0
        print('reset! Ep:', self.ep_counter)
        global soft_body_model_vertices
        soft_body_model_vertices = np.load(current_dir + "/initial_vertices.npy")
        return self.state

    def step(self, action):
        new_state = self.state + action
        np.clip(new_state, self.observation_space.low, self.observation_space.high, out=new_state)
        self.traj.append(np.array(new_state))
        traj_points = np.vstack((self.state, new_state))
        traj_points = interpolate_trajectory(traj_points, traj_length)

        global grasp_seq, soft_body_model_vertices, grasp_point_pos
        grasp_seq = grasp_seq.reshape(traj_length, 1, 3) * scene_scaling
        if self.step_counter > 0:
            soft_body_model_vertices = np.load(tmp_vertices_file)
        else:
            soft_body_model_vertices = np.load(initial_vertices_file)
        grasp_point_pos = grasp_seq

        initialize_objects()
        animation(total_time)
        point_cloud_observation = np.nan_to_num(vertex_x.to_numpy())
        loss = compute_chamfer_distance(target, point_cloud_observation)
        self.cd = loss
        a = 1
        b = -1
        reward = a - b * loss
        self.reward_for_ep += reward
        print(f"Ep：{self.ep_counter}, Step: {self.step_counter}, Loss:{self.cd}, Reward: {reward}, Reward for ep: {self.reward_for_ep}")

        if self.step_counter == self.max_steps - 1:
            if self.ep_counter not in self.step_rewards:
                self.step_rewards[self.ep_counter] = self.reward_for_ep
                self.ep_loss.append(self.cd)
                self.reward_arr.append(self.reward_for_ep)
                self.eps_traj.append(np.array(self.traj))

        done = self.step_counter >= self.max_steps - 1
        if self.step_counter >= self.max_steps - 1:
            self.step_counter = 0
        else:
            self.step_counter += 1
            self.state = new_state
        info = {}
        return point_cloud_observation, new_state, reward, done, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

def make_env(n):
    def _init():
        env = CustomEnv(n)
        return env
    return _init

class StopOnDoneCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(StopOnDoneCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        done = self.locals.get("done")
        if done:
            return False
        return True

def run_experiment(seed):
    np.random.seed(seed)
    env = CustomEnv(n_ep)
    env.seed(seed)
    model = SAC("MlpPolicy", env, verbose=1, seed=seed)
    callback = StopOnDoneCallback()
    model.learn(total_timesteps=n_step_per_ep * n_ep, callback=callback)
    return env.step_rewards, env.reward_arr, env.ep_loss

if __name__ == "__main__":
    tmp_dir = os.path.abspath(os.path.dirname(__file__)) + '/../tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    tmp_vertices_file = tmp_dir + "/tmp_vertices.npy"
    initial_vertices_file = tmp_dir + "/initial_vertices.npy"
    seeds = list(np.random.randint(10000, size=5))
    chamfer_distances_for_all_ep = []
    for seed in seeds:
        print(f"Running experiment with seed {seed}")
        _, _, chamfer_distance = run_experiment(int(seed))
        chamfer_distances_for_all_ep.append(chamfer_distance)
    np.save(result_dir + "/RL_chamfer_distances.npy", np.array(chamfer_distances_for_all_ep))

