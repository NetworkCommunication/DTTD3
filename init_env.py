import numpy as np


def generate_uav_program_list(cache_size, num_uavs, total_task_types):
    uav_cache_list = np.zeros((num_uavs, total_task_types))
    for i in range(num_uavs):
        cache_indices = np.random.choice(total_task_types, cache_size, replace=False)
        for idx in cache_indices:
            uav_cache_list[i][idx] = 1
    return uav_cache_list


def compute_program_storage(program_size,uav_cache_list):
    sum_size = 0
    for i in uav_cache_list:
        if uav_cache_list[i] == 1:
            sum_size += program_size[i]
    return sum_size

def calculate_distance(user_positions, drone_position):
    distances = np.linalg.norm(user_positions - drone_position, axis=1)
    return distances


