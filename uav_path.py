import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# 设置随机种子
np.random.seed(5)

# 参数设置
area_size = 200  # 区域大小
num_users = 20  # 用户数量
num_uavs = 3  # 无人机数量
coverage_radius = 50  # 覆盖半径
num_time_slots = 100  # 时隙数量
user_move_distance = 5  # 用户移动距离

# 初始化无人机位置
uav_positions = np.array([
    [50, 125],
    [125, 150],
    [100, 50]
])

# 随机生成用户初始位置
user_positions = np.random.rand(num_users, 2) * area_size

# 定义函数：计算每个无人机覆盖的用户情况
def calculate_coverage(uav_positions, user_positions, radius):
    distances = cdist(uav_positions, user_positions)  # 计算每个无人机到每个用户的距离
    coverage = distances <= radius  # 判断距离是否在覆盖半径内
    return coverage

# 定义函数：更新用户位置
def update_user_positions(user_positions, move_distance, area_size):
    move = np.random.choice([-1, 1], size=(user_positions.shape[0], 2)) * move_distance  # 随机生成移动向量
    user_positions += move  # 更新用户位置
    user_positions = np.clip(user_positions, 0, area_size)  # 限制用户位置在区域内
    return user_positions

# 模拟多个时隙
all_uav_positions = [uav_positions.copy()]  # 存储所有时隙的无人机位置
all_coverages = []  # 存储每个时隙每个无人机的覆盖用户数量

for t in range(num_time_slots):
    # 更新用户位置
    user_positions = update_user_positions(user_positions, user_move_distance, area_size)

    # 计算当前覆盖的用户数量
    max_covered = 0  # 最大覆盖用户数量
    best_positions = uav_positions.copy()  # 最佳无人机位置

    # 尝试移动无人机，选择最优位置
    for i, uav in enumerate(uav_positions):
        for dx in range(-20, 21):
            for dy in range(-20, 21):
                if dx == 0 and dy == 0:
                    continue
                new_position = uav + np.array([dx, dy])  # 计算新的无人机位置
                new_position = np.clip(new_position, 0, area_size)  # 限制无人机位置在区域内
                temp_positions = uav_positions.copy()
                temp_positions[i] = new_position  # 临时存储移动后的无人机位置

                # 计算覆盖的用户数量，确保不重复覆盖
                coverage = calculate_coverage(temp_positions, user_positions, coverage_radius)  # 计算覆盖情况
                unique_coverage = np.any(coverage, axis=0)  # 去除重复覆盖
                covered = np.sum(unique_coverage)  # 计算覆盖用户数量

                if covered > max_covered:
                    max_covered = covered
                    best_positions = temp_positions.copy()  # 更新最佳无人机位置

    # 更新无人机位置
    uav_positions = best_positions
    all_uav_positions.append(uav_positions.copy())  # 存储当前时隙的无人机位置

    # 计算当前每个无人机覆盖的用户数量
    coverage = calculate_coverage(uav_positions, user_positions, coverage_radius)  # 计算覆盖情况
    individual_coverage = np.sum(coverage, axis=1)  # 计算每个无人机覆盖的用户数量
    all_coverages.append((individual_coverage, coverage))  # 存储每个无人机覆盖的用户数量和覆盖情况

# 转换覆盖数据为numpy数组
all_coverages = [(ind_coverage, coverage) for ind_coverage, coverage in all_coverages]

# 可视化路径和最终覆盖范围
plt.figure(figsize=(8, 8))
colors = ['red', 'green', 'blue']

for i in range(num_uavs):
    uav_path = np.array([pos[i] for pos in all_uav_positions])  # 获取每个无人机的轨迹
    plt.plot(uav_path[:, 0], uav_path[:, 1], label=f'UAV {i + 1} Path', color=colors[i])  # 绘制无人机轨迹
    plt.scatter(uav_path[:, 0], uav_path[:, 1], s=20, color=colors[i])  # 绘制无人机位置点
    plt.scatter(uav_path[-1, 0], uav_path[-1, 1], s=100, color='black', marker='x')  # 绘制最终无人机位置
    circle = plt.Circle((uav_path[-1, 0], uav_path[-1, 1]), coverage_radius, color=colors[i], alpha=0.3)  # 绘制覆盖范围
    plt.gca().add_patch(circle)

plt.scatter(user_positions[:, 0], user_positions[:, 1], c='blue', label='Users')  # 绘制用户位置
plt.xlim(0, area_size)
plt.ylim(0, area_size)
plt.legend()
plt.title('UAV Paths and Final Coverage Areas')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.grid(True)
plt.show()

# 输出每个时隙每个无人机覆盖的用户数量和用户编号
for t in range(num_time_slots):
    ind_coverage, coverage = all_coverages[t]
    print(f'Time Slot {t + 1}: UAV coverages: {ind_coverage}')
    for i in range(num_uavs):
        covered_users = np.where(coverage[i])[0]
        print(f'  UAV {i + 1} covers users: {covered_users}')

