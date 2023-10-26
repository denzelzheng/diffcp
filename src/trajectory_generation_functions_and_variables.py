import numpy as np
from scipy.interpolate import interp1d
import scipy.spatial
from src.initialization import *

def compute_chamfer_distance(array1, array2):
    chamfer_distance = scipy.spatial.distance.cdist(array1, array2)
    chamfer_distance = np.min(chamfer_distance, axis=1)
    chamfer_distance = np.mean(chamfer_distance)
    return chamfer_distance

def interpolate_trajectory(trajectory, total_length):

    original_length = len(trajectory)
    old_indices = np.linspace(0, original_length-1, original_length)
    new_indices = np.linspace(0, original_length-1, total_length)
    interp_funcs = [interp1d(old_indices, trajectory[:, i], kind='linear') for i in range(3)]
    new_trajectory = np.vstack([f(new_indices) for f in interp_funcs]).T
    return new_trajectory

target = np.load(data_dir + '/target_point_cloud.npy')