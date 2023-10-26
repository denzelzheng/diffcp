import subprocess
import numpy as np
import os
import gym
from stable_baselines3 import SAC
from gym import spaces
from stable_baselines3.common.callbacks import BaseCallback
from scipy.interpolate import interp1d

def interpolate_trajectory(trajectory, total_length):
    original_length = len(trajectory)
    old_indices = np.linspace(0, original_length-1, original_length)
    new_indices = np.linspace(0, original_length-1, total_length)
    interp_funcs = [interp1d(old_indices, trajectory[:, i], kind='linear') for i in range(3)]
    new_trajectory = np.vstack([f(new_indices) for f in interp_funcs]).T
    return new_trajectory

class CustomEnv(gym.Env):
    def __init__(self, n):
        self.n = n
        low = np.array([0.3, 0, -0.5])
        high = np.array([1, 0.45, 0.5])
        action_range = 2.1 / n
        self.observation_space = spaces.Box(low=low, high=high, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(low=-action_range, high=action_range, shape=(3,), dtype=np.float32)
        self.step_counter = 0
        self.ep_counter = -1
        self.max_steps = n
        self.step_rewards = {}
        self.max_ep = 2
        self.reward_for_ep = 0

    def reset(self):
        self.state = np.array([0.8, 0, -0.3])
        self.step_counter = 0
        self.ep_counter += 1
        self.reward_for_ep = 0
        self.chamfer_distance_for_ep = 0
        print('reset! Ep:', self.ep_counter)
        return self.state

    def step(self, action):
        point_cloud_observation = np.load(tmp_dir + '/point_cloud_observation.npy')
        new_state = self.state + action
        np.clip(new_state, self.observation_space.low, self.observation_space.high, out=new_state)
        traj_points = np.vstack((self.state, new_state))
        traj_points = interpolate_trajectory(traj_points, 31)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        np.save(tmp_dir + "/RL_gen_traj.npy", traj_points)

        subprocess.run(["python", current_dir + "/trajectory_sim_RL.py", str(self.step_counter), str(self.n)], capture_output=True, text=True)
        file_path = tmp_dir + "/RL_chamfer.txt"

        with open(file_path, "r") as f:
            content = f.read()
        loss = float(content)
        a = 1
        b = -1
        reward = a - b * loss
        self.reward_for_ep += reward
        print(f"Step: {self.step_counter}, Reward: {reward}, Reward for ep: {self.reward_for_ep}")

        if self.step_counter == self.max_steps - 1:
            if self.ep_counter not in self.step_rewards:
                self.step_rewards[self.ep_counter] = self.reward_for_ep
                self.chamfer_distance_for_ep[self.ep_counter] = loss

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
    env = CustomEnv(n_eps)
    env.seed(seed)
    model = SAC("MlpPolicy", env, verbose=1, seed=seed)
    callback = StopOnDoneCallback()
    model.learn(total_timesteps=n_eps * steps_per_ep, callback=callback)
    return env.chamfer_distance_for_ep

if __name__ == "__main__":
    tmp_dir = os.path.abspath(os.path.dirname(__file__)) + '/../tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    steps_per_ep = 50
    n_eps = 31
    seeds = list(np.random.randint(10000, size=5))
    chamfer_distances_for_all_ep = []
    for seed in seeds:
        print(f"Running experiment with seed {seed}")
        chamfer_distance = run_experiment(int(seed))
        chamfer_distances_for_all_ep.append(chamfer_distance)
    np.save("chamfer_distances.npy", np.array(chamfer_distances_for_all_ep))
