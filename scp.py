import numpy as np
from collections import Counter

# 任务类型总数
TOTAL_TASK_TYPES = 10
# 地面设备数量
NUM_DEVICES = 20
# 基站缓存的程序数量
CACHE_SIZE = 5
# 一个周期的时隙数
SLOT_CYCLE = 10
# Zipf分布参数
ZIPF_PARAM = 1.1


# 生成任务类型，根据Zipf分布，并限制任务类型在1到TOTAL_TASK_TYPES之间
def generate_tasks(num_devices, num_task_types, zipf_param):
    tasks = np.random.zipf(zipf_param, num_devices)
    tasks = np.mod(tasks - 1, num_task_types) + 1
    return tasks


# 计算任务类型的权重，根据zipf分布和服务的任务次数
def calculate_task_weights(task_log, num_task_types, zipf_weights, service_weights):
    weighted_counts = np.zeros(num_task_types + 1)

    for task_type in range(1, num_task_types + 1):
        zipf_weight = zipf_weights[task_type - 1]
        service_weight = service_weights[task_type - 1]  # 直接使用服务权重，不需要从task_counter获取
        weighted_counts[task_type] = zipf_weight + service_weight

    return weighted_counts


# 更新缓存，选择新的缓存程序类型
def update_cache(task_weights, cache_size):
    most_common_tasks = np.argsort(task_weights)[-cache_size:][::-1]
    return most_common_tasks.tolist()


# 模拟基站服务和缓存调整
def simulate_service(num_devices, num_task_types, cache_size, zipf_param, slot_cycle, total_slots):
    # 初始化缓存程序
    current_cache = list(np.random.choice(range(1, num_task_types + 1), cache_size, replace=False))
    task_log = []

    # 计算zipf分布的权重
    zipf_weights = np.random.zipf(zipf_param, num_task_types)
    zipf_weights = np.mod(zipf_weights - 1, num_task_types) + 1

    for slot in range(total_slots):
        # 生成每个设备的任务
        tasks = generate_tasks(num_devices, num_task_types, zipf_param)

        # 随机选择一个设备进行服务
        device_index = np.random.randint(num_devices)
        task_type = tasks[device_index]

        # 记录任务类型
        task_log.append(task_type)

        # 输出当前时隙的服务信息
        print(f"时隙 {slot + 1}: 设备 {device_index + 1} 产生任务类型 {task_type}, 当前缓存: {current_cache}")

        # 每 slot_cycle 个时隙更新一次缓存
        if (slot + 1) % slot_cycle == 0:
            # 计算服务次数的权重
            service_weights = [task_log.count(i) for i in range(1, num_task_types + 1)]
            task_weights = calculate_task_weights(task_log[-slot_cycle:], num_task_types, zipf_weights, service_weights)
            current_cache = update_cache(task_weights, cache_size)
            print(f"更新缓存: {current_cache}")


# 总共模拟的时隙数
TOTAL_SLOTS = 100

simulate_service(NUM_DEVICES, TOTAL_TASK_TYPES, CACHE_SIZE, ZIPF_PARAM, SLOT_CYCLE, TOTAL_SLOTS)

