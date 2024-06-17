import numpy as np
from collections import Counter
import random
import os
import math
from scipy.spatial.distance import cdist
import init_env

# 任务类型总数
TOTAL_TASK_TYPES = 10
# 地面设备数量
NUM_DEVICES = 20
# 初始缓存的程序数量
CACHE_SIZE = 5
# 缓存更新的时隙数
SLOT_CYCLE = 10
# Zipf分布参数
ZIPF_PARAM = 1.1

# 生成任务类型，根据Zipf分布，并限制任务类型在1到TOTAL_TASK_TYPES之间
def generate_tasks(num_devices, num_task_types, zipf_param):
    tasks = np.random.zipf(zipf_param, num_devices)
    tasks = np.mod(tasks - 1, num_task_types) + 1
    return tasks

# 计算任务类型的权重，根据zipf分布和被服务的任务次数
def calculate_task_weights(task_log, num_task_types, zipf_weights, service_weights):
    task_counter = Counter(task_log)
    weighted_counts = np.zeros(num_task_types)

    for task_type in range(1, num_task_types + 1):
        zipf_weight = zipf_weights[task_type - 1]
        service_weight = service_weights[task_type - 1] if task_type in task_counter else 0
        weighted_counts[task_type - 1] = zipf_weight + service_weight

    return weighted_counts

# 更新缓存，选择新的缓存程序类型
def update_cache(task_weights, cache_size, program_size, cache_capacity):
    sorted_indices = np.argsort(-task_weights)  # 按权重降序排序
    current_cache = []
    current_size = 0.0

    for index in sorted_indices:
        if task_weights[index] == 0:
            continue  # 跳过权重为0的任务
        if current_size + program_size[index] <= cache_capacity:
            current_cache.append(index)
            current_size += program_size[index]
        else:
            # 尝试替换较小的程序
            for i, cached_index in enumerate(current_cache):
                if program_size[cached_index] > program_size[index] and \
                   current_size - program_size[cached_index] + program_size[index] <= cache_capacity:
                    current_cache[i] = index
                    current_size = current_size - program_size[cached_index] + program_size[index]
                    break

    return current_cache

# 计算每个无人机覆盖的用户情况
def calculate_coverage(uav_positions, user_positions, radius):
    distances = cdist(uav_positions, user_positions)  # 计算每个无人机到每个用户的距离
    coverage = distances <= radius  # 判断距离是否在覆盖半径内
    return coverage

# 更新用户位置
def update_user_positions(user_positions, move_distance, area_size):
    move = np.random.choice([-1, 1], size=(user_positions.shape[0], 2)) * move_distance  # 随机生成移动向量
    user_positions += move  # 更新用户位置
    user_positions = np.clip(user_positions, 0, area_size)  # 限制用户位置在区域内
    return user_positions

# 参数设置
area_size = 200  # 区域大小
num_users = NUM_DEVICES  # 用户数量
num_uavs = 3  # 无人机数量
coverage_radius = 50  # 覆盖半径
num_time_slots = 100  # 时隙数量
user_move_distance = 5  # 用户移动距离
uav_move_distance = 10  # 无人机移动距离

# 初始化无人机位置
uav_positions = np.array([
    [50, 125],
    [125, 150],
    [100, 50]
])

# 随机生成用户初始位置
user_positions = np.random.rand(num_users, 2) * area_size

# 初始化MultiUAVEnv环境
class MultiUAVEnv(object):
    def __init__(self):
        self.uav_num = num_uavs
        self.ue_num = num_users
        self.uav_location = uav_positions
        self.area_size = area_size
        self.ue_location_list = user_positions
        self.uav_height = 100  # UAV悬停高度
        self.uav_range = 50  # UAV通信覆盖范围
        self.initial_uav_energy = 1000  # 初始UAV能量
        self.uav_energy = np.array([self.initial_uav_energy] * self.uav_num)  # UAV能量
        self.uav_f = 10e9  # UAV计算频率
        self.origin_uav_storage = 1.0 * 1048576  # UAV存储空间
        self.task_num = TOTAL_TASK_TYPES
        self.program_size = np.random.uniform(0.1, 0.2, TOTAL_TASK_TYPES)  # 程序大小在 [0.1, 0.2] Mbits 之间
        self.uav_cache_capacity = 1.0  # 无人机缓存容量为 1 Mbits
        self.uav_cache_list = init_env.generate_uav_program_list(CACHE_SIZE, self.uav_num, self.task_num)
        self.ue_f = 0.5e9  # UE计算频率
        self.ue_tranmission_power = 0.1  # UE传输功率
        self.ue_program_id_list = generate_tasks(num_users, TOTAL_TASK_TYPES, ZIPF_PARAM)
        self.ground_length = area_size
        self.ground_width = area_size
        self.channel_gain = 1e-5
        self.total_bandwidth = 2e7
        self.noise_power_density = 10 ** (-13)
        self.W_list = np.random.randint(0.2e9, 1e9, size=self.ue_num)
        self.L_list = np.random.uniform(104857.6, 209715.2, self.ue_num)
        self.T = 200
        self.delta_t = 2
        self.slot_num = int(self.T / self.delta_t)
        self.coefficient = 10e-27
        self.wireless_backhaul_rate = 1e6
        self.R_b = 3e5
        self.action_bound = [-1, 1]
        self.action_dim = self.uav_num * 3  # 移除 need_cache 动作，动作维度变为 3
        self.state_dim = self.ue_num * 4 + (3 + self.task_num) * self.uav_num
        self.start_state = self.get_initial_state()
        self.state = self.start_state
        self.task_log = []
        self.service_history = np.zeros((self.uav_num, self.ue_num))  # 添加记录服务历史的矩阵

        # 计算zipf分布的权重
        self.zipf_weights = np.random.zipf(ZIPF_PARAM, TOTAL_TASK_TYPES)
        self.zipf_weights = np.mod(self.zipf_weights - 1, TOTAL_TASK_TYPES) + 1

    def get_initial_state(self):
        state = np.ravel(self.uav_energy)  # UAV能量
        state = np.append(state, np.ravel(self.uav_location))  # UAV位置
        state = np.append(state, np.ravel(self.uav_cache_list))  # UAV所具备的程序
        state = np.append(state, np.ravel(self.ue_location_list))  # UE的位置
        state = np.append(state, self.W_list)  # UE生成任务的大小
        state = np.append(state, self.ue_program_id_list)  # UE生成任务的类型
        return state

    def reset(self):
        self.start_state = self.get_initial_state()
        return self._get_obs()

    def _get_obs(self):
        return self.start_state

    def reset_env(self):
        self.state = self.get_initial_state()
        self.uav_energy = np.array([self.initial_uav_energy] * self.uav_num)  # 重置UAV能量
        self.ue_location_list = np.random.randint(0, 201, size=[self.ue_num, 2])  # UE初始位置
        self.uav_cache_list = init_env.generate_uav_program_list(CACHE_SIZE, self.uav_num, self.task_num)
        self.service_history = np.zeros((self.uav_num, self.ue_num))  # 重置服务历史
        self.reset_step()

    def reset_step(self):
        self.L_list = np.random.uniform(104857.6, 209715.2, self.ue_num)  # 每个任务上传的数据大小
        self.W_list = np.random.randint(0.2e9, 1e9, size=self.ue_num)
        self.ue_program_id_list = generate_tasks(self.ue_num, TOTAL_TASK_TYPES, ZIPF_PARAM)

    def step(self, action, step):
        action = (action + 1) / 2  # 将取值区间位-1~1的action -> 0~1的action。 避免原来action_bound为[0,1]时训练actor网络tanh函数一直取边界0
        total_reward = 0

        # 计算无人机的覆盖情况
        coverage = calculate_coverage(self.uav_location, self.ue_location_list, self.uav_range)
        user_distances = cdist(self.uav_location, self.ue_location_list)  # 计算每个无人机到每个用户的距离

        selected_users = set()  # 用于存储已被选择的用户

        for uav_index in range(self.uav_num):
            # 获取当前无人机覆盖的用户
            covered_users = np.where(coverage[uav_index])[0]

            if len(covered_users) > 0:
                # 优先选择覆盖范围内的用户
                ue_index = int(action[uav_index * 3] * (len(covered_users) - 1))  # 在覆盖范围内的用户中选择
                ue_index = covered_users[ue_index]
            else:
                # 如果没有覆盖范围内的用户，则随机选择一个用户进行本地计算
                ue_index = np.random.randint(0, self.ue_num)

            # 确保选择的用户是唯一的
            while ue_index in selected_users:
                ue_index = (ue_index + 1) % self.ue_num

            selected_users.add(ue_index)

            # 检查用户是否在覆盖范围内
            in_coverage = ue_index in covered_users

            # 判断是否卸载任务
            is_offloading = 1 if action[uav_index * 3 + 1] >= 0.5 else 0
            computation_ratio = action[uav_index * 3 + 2] if action[uav_index * 3 + 2] != 0 else 0.01
            uav_energy_consumption = 0.01 * self.coefficient * self.uav_f ** 2 * self.W_list[ue_index] * computation_ratio

            if step % SLOT_CYCLE == 0:
                # 计算服务次数的权重
                service_weights = [self.task_log.count(i) for i in range(1, TOTAL_TASK_TYPES + 1)]
                task_weights = calculate_task_weights(self.task_log[-SLOT_CYCLE:], TOTAL_TASK_TYPES, self.zipf_weights,
                                                      service_weights)
                most_common_tasks = update_cache(task_weights, CACHE_SIZE, self.program_size, self.uav_cache_capacity)
                self.uav_cache_list[uav_index] = np.zeros(TOTAL_TASK_TYPES)
                for task in most_common_tasks:
                    if task >= 0:
                        self.uav_cache_list[uav_index][task] = 1

            task_type = self.ue_program_id_list[ue_index]
            has_cache = self.uav_cache_list[uav_index][task_type - 1] == 1

            # 计算奖励
            if is_offloading:
                uav_delay = self.com_delay(ue_index, computation_ratio)
                bs_delay = self.base_station_delay(ue_index)
                if uav_delay < bs_delay and has_cache and self.uav_energy[uav_index] > uav_energy_consumption:
                    # 卸载到无人机
                    delay = uav_delay
                    delay_source = '无人机'
                    self.uav_energy[uav_index] -= uav_energy_consumption
                else:
                    # 卸载到基站
                    delay = bs_delay
                    delay_source = '基站'
            else:
                # 本地计算
                local_delay = self.local_delay(ue_index)
                delay = local_delay
                delay_source = '本地'

            reward = -delay
            # 输出调试信息
            #print(
            #    f"无人机 {uav_index} 选择用户 {ue_index}，覆盖范围内: {in_coverage}，是否卸载: {is_offloading}，是否具有缓存: {has_cache}，计算时延来源: {delay_source}")

            self.reset2(ue_index, computation_ratio, delay, uav_index)
            total_reward += reward

            # 记录任务类型
            self.task_log.append(self.ue_program_id_list[ue_index])

        # 更新用户位置
        self.ue_location_list = update_user_positions(self.ue_location_list, user_move_distance, area_size)

        # 更新无人机位置
        max_covered = 0  # 最大覆盖用户数量
        best_positions = self.uav_location.copy()  # 最佳无人机位置

        # 尝试移动无人机，选择最优位置
        for i, uav in enumerate(self.uav_location):
            for move_x in [-1, 0, 1]:
                for move_y in [-1, 0, 1]:
                    if move_x == 0 and move_y == 0:
                        continue
                    new_position = uav + np.array([move_x, move_y]) * uav_move_distance  # 计算新的无人机位置
                    new_position = np.clip(new_position, 0, area_size)  # 限制无人机位置在区域内
                    temp_positions = self.uav_location.copy()
                    temp_positions[i] = new_position  # 临时存储移动后的无人机位置

                    # 计算覆盖的用户数量，确保不重复覆盖
                    coverage = calculate_coverage(temp_positions, self.ue_location_list, coverage_radius)  # 计算覆盖情况
                    unique_coverage = np.any(coverage, axis=0)  # 去除重复覆盖
                    covered = np.sum(unique_coverage)  # 计算覆盖用户数量

                    if covered > max_covered:
                        max_covered = covered
                        best_positions = temp_positions.copy()  # 更新最佳无人机位置

        self.uav_location = best_positions

        average_reward = total_reward / self.uav_num
        next_state = self._get_obs()
        done = False  # 你可以根据你的环境定义条件来设置done
        info = {}  # 你可以在这里添加任何调试信息

        return next_state, average_reward, done, info

    def reset2(self, ue_index, computation_ratio, delay, uav_index):
        self.reset_step()
        file_name = 'output.txt'
        with open(file_name, 'a') as file_obj:
            file_obj.write("\nUAV-" + '{:d}'.format(uav_index) + ", UE-" + '{:d}'.format(
                ue_index) + ", task size: " + '{:.5f}'.format(
                self.W_list[ue_index]) + ", computation ratio:" + '{:.5f}'.format(computation_ratio))
            file_obj.write("\ndelay:" + '{:.5f}'.format(delay))

    def com_delay(self, ue_index, computation_ratio):
        transition_rate = self.total_bandwidth * math.log2(1 + self.ue_tranmission_power * self.channel_gain /
                        (self.total_bandwidth * self.noise_power_density))
        tran_delay = self.L_list[ue_index] / transition_rate
        com_delay = self.W_list[ue_index] / (self.uav_f * computation_ratio)
        delay = tran_delay + com_delay
        return delay

    def local_delay(self, ue_index):
        local_delay = self.W_list[ue_index] / self.ue_f
        return local_delay

    def base_station_delay(self, ue_index):
        transmission_delay = self.L_list[ue_index] / self.R_b
        return transmission_delay

# 初始化环境
env = MultiUAVEnv()
