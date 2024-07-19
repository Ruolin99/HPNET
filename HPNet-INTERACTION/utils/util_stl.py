# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:29:25 2024

@author: Ruolin Shi
"""

import torch
import math
from typing import Tuple
torch.set_float32_matmul_precision('high')


def generate_target(position: torch.Tensor, mask: torch.Tensor, num_historical_steps: int, num_future_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
    target_traj = [position[:, i + 1:i + 1 + num_future_steps] for i in range(num_historical_steps)]
    target_traj = torch.stack(target_traj, dim=1)
    target_mask = [mask[:, i + 1:i + 1 + num_future_steps] for i in range(num_historical_steps)]
    target_mask = torch.stack(target_mask, dim=1)
    return target_traj, target_mask


def calculate_vehicle_ttc(car_positions, car_velocities):
    """
    计算车辆之间的TTC

    参数：
    car_positions: (torch.Tensor) 车辆位置张量，维度为 (num_cars, H, F, 2)
    car_velocities: (torch.Tensor) 车辆速度张量，维度为 (num_cars, H, F, 2)

    返回：
    vehicle_ttc: (torch.Tensor) 车辆之间的TTC张量，维度为 (num_cars, num_cars, H, F)
    """
    num_cars = car_positions.shape[0]
    H = car_positions.shape[1]
    F = car_positions.shape[2]

    vehicle_ttc = torch.full((num_cars, num_cars, H, F), 1000.0, device='cuda')

    for i in range(num_cars):
        for j in range(num_cars):
            if i != j:
                for t in range(H):
                    for f in range(F):
                        o_k = car_positions[i, t, f]
                        o_m = car_positions[j, t, f]
                        v_k = car_velocities[i, t, f]
                        v_m = car_velocities[j, t, f]
                        delta_pos = o_k - o_m
                        delta_vel = v_k - v_m

                        # 计算车辆之间的TTC
                        relative_speed_sq = torch.dot(delta_vel, delta_vel)
                        if relative_speed_sq != 0:
                            ttc1 = -torch.dot(delta_pos, delta_vel) / relative_speed_sq
                            if ttc1 > 0:
                                vehicle_ttc[i, j, t, f] = ttc1

    return vehicle_ttc

def process_ettc_ttcx_ttcy_ttcyy(ettc, ttcx, ttcy, ttcyy):
    """
    处理ETTC、TTCx、TTcy和TTCyy的计算结果

    参数：
    ettc: (torch.Tensor) ETTC张量，维度为 (num_cars, num_others, H, F)
    ttcx: (torch.Tensor) TTCx张量，维度为 (num_cars, num_others, H, F)
    ttcy: (torch.Tensor) TTcy张量，维度为 (num_cars, num_others, H, F)
    ttcyy: (torch.Tensor) TTCyy张量，维度为 (num_cars, num_others, H, F)

    返回：
    sum_min: (torch.Tensor) 每辆车的最小ETTC，维度为 (num_cars, H)
    """
    min_ettc, _ = torch.min(ettc, dim=1)  # 维度 (num_cars, H, F)
    min_ttcx, _ = torch.min(ttcx, dim=1)  # 维度 (num_cars, H, F)
    min_ttcy, _ = torch.min(ttcy, dim=1)  # 维度 (num_cars, H, F)
    min_ttcyy, _ = torch.min(ttcyy, dim=1)  # 维度 (num_cars, H, F)

    min_ettc[min_ettc == float('inf')] = 1000
    min_ttcx[min_ttcx == float('inf')] = 1000
    min_ttcy[min_ttcy == float('inf')] = 1000
    min_ttcyy[min_ttcyy == float('inf')] = 1000

    min_ttc = torch.min(torch.min(min_ettc, min_ttcx), torch.min(min_ttcy, min_ttcyy))  # 维度 (num_cars, H, F)
    sum_min, _ = torch.min(min_ttc, dim=-1)  # 维度 (num_cars, H)

    return sum_min



def calculate_ettc(car_positions, car_velocities, others_positions, others_velocities):
    """
    计算ETTC、TTCx、TTcy和TTCyy

    参数：
    car_positions: (torch.Tensor) 车辆位置张量，维度为 (num_cars, H, F, 2)
    car_velocities: (torch.Tensor) 车辆速度张量，维度为 (num_cars, H, F, 2)
    others_positions: (torch.Tensor) 其他交通参与者位置张量，维度为 (num_others, H, F, 2)
    others_velocities: (torch.Tensor) 其他交通参与者速度张量，维度为 (num_others, H, F, 2)

    返回：
    sum_min: (torch.Tensor) 每辆车的最小ETTC，维度为 (num_cars, H, F, 2)
    min_vehicle_ttc: (torch.Tensor) 每辆车的最小TTC，维度为 (num_cars, H, F, 2)
    """
    num_cars = car_positions.shape[0]
    num_others = others_positions.shape[0]
    H = car_positions.shape[1]
    F = car_positions.shape[2]

    if num_others != 0:
        # 初始化ETTC、TTCx、TTcy和TTCyy张量
        ettc = torch.full((num_cars, num_others, H, F), 1000.0 ,device='cuda')
        ttcx = torch.full((num_cars, num_others, H, F), 1000.0,device='cuda')
        ttcy = torch.full((num_cars, num_others, H, F), 1000.0,device='cuda')
        ttcyy = torch.full((num_cars, num_others, H, F), 1000.0,device='cuda')

        # 计算ETTC、TTCx、TTcy和TTCyy
        for i in range(num_cars):
            for j in range(num_others):
                for t in range(H):
                    for f in range(F):
                        o_k = car_positions[i, t, f]
                        o_m = others_positions[j, t, f]
                        v_k = car_velocities[i, t, f]
                        v_m = others_velocities[j, t, f]
                        delta_pos = o_k - o_m
                        delta_vel = v_k - v_m

                        # 计算 ETTC
                        relative_speed_sq = torch.dot(delta_vel, delta_vel)
                        if relative_speed_sq != 0:
                            ettc1 = -torch.dot(delta_pos, delta_vel) / relative_speed_sq
                            if ettc1 > 0:
                                ettc[i, j, t, f] = ettc1

                        # 计算 TTCx 和 TTCy
                        if delta_vel[0] != 0:
                            ttcx1 = delta_pos[0] / delta_vel[0]
                            if ttcx1 > 0:
                                ttcx[i, j, t, f] = ttcx1
                        if delta_vel[1] != 0:
                            ttcy1 = delta_pos[1] / delta_vel[1]
                            if ttcy1 > 0:
                                ttcy[i, j, t, f] = ttcy1

                        # 计算 TTCYY
                        relative_speed = torch.norm(delta_vel)
                        if relative_speed != 0:
                            ttcyy1 = torch.norm(delta_pos) / relative_speed
                            if ttcyy1 > 0:
                                ttcyy[i, j, t, f] = ttcyy1

        # 处理ETTC、TTCx、TTcy和TTCyy，计算最小值
        sum_min = process_ettc_ttcx_ttcy_ttcyy(ettc, ttcx, ttcy, ttcyy)
    else:
        ettc = torch.full((num_cars, H, F), 1000.0)
        ttcx = torch.full((num_cars, H, F), 1000.0)
        ttcy = torch.full((num_cars, H, F), 1000.0)
        ttcyy = torch.full((num_cars, H, F), 1000.0)
        sum_min = process_ettc_ttcx_ttcy_ttcyy(ettc.unsqueeze(1), ttcx.unsqueeze(1), ttcy.unsqueeze(1), ttcyy.unsqueeze(1))

    # 计算车辆之间的TTC
    vehicle_ttc = calculate_vehicle_ttc(car_positions, car_velocities)

    # 计算每辆车的最小TTC
    min_vehicle_ttc = torch.min(vehicle_ttc, dim=1)[0].min(dim=-1)[0]  # 维度 (num_cars, H, F, 2)

    return sum_min, min_vehicle_ttc





def make_torch_tensor(arr, keep_1dim=True):
    if isinstance(arr, torch.Tensor):
        return arr
    arr = torch.tensor(arr, dtype=torch.float32)
    if keep_1dim and arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

def calculate_robustness_d(traj_in,traj_oc_in):
    traj_in = make_torch_tensor(traj_in, keep_1dim=False)
    traj_oc_in = make_torch_tensor(traj_oc_in, keep_1dim=False)

    r_ox = torch.abs(traj_in[:, 0] - traj_oc_in[:, 0])
    r_oy = torch.abs(traj_in[:, 1] - traj_oc_in[:, 1])
    return torch.min(r_ox), torch.min(r_oy)

def calculate_robustness_car(traj_in, size_in, traj_oc_in, size_oc_in):
    """
    计算两辆车之间的鲁棒性。

    参数：
    traj_in: (torch.Tensor) 目标车辆的轨迹，维度为 (num_steps, 2)
    size_in: (torch.Tensor) 目标车辆的尺寸，维度为 (2,)
    traj_oc_in: (torch.Tensor) 另一辆车的轨迹，维度为 (num_steps, 2)
    size_oc_in: (torch.Tensor) 另一辆车的尺寸，维度为 (2,)

    返回：
    min_r_ox: (float) 目标车辆与另一辆车之间 x 方向最小的碰撞鲁棒性
    min_r_oy: (float) 目标车辆与另一辆车之间 y 方向最小的碰撞鲁棒性
    """
    traj_in = make_torch_tensor(traj_in, keep_1dim=False)
    size_in = make_torch_tensor(size_in, keep_1dim=True)
    traj_oc_in = make_torch_tensor(traj_oc_in, keep_1dim=False)
    size_oc_in = make_torch_tensor(size_oc_in, keep_1dim=True)

    r_ox = torch.abs(traj_in[:, 0] - traj_oc_in[:, 0]) - size_in[0] / 2 - size_oc_in[0] / 2
    r_oy = torch.abs(traj_in[:, 1] - traj_oc_in[:, 1]) - size_in[1] / 2 - size_oc_in[1] / 2
    return torch.min(r_ox), torch.min(r_oy)

def compute_robustness_collision(raw, trajectory_proposed, data):
    """
    计算车辆与车辆、车辆与其他交通参与者之间的碰撞鲁棒性

    参数：
    raw: (torch.Tensor) 其他交通参与者的轨迹数据，维度为 (n, H, F, 2)
    trajectory_proposed: (torch.Tensor) 车辆提议的轨迹数据，维度为 (n, H, F, 2)
    data: (dict) 包含车辆和其他交通参与者信息的字典，包含 'agent' 字段

    返回：
    min_robustness_car_car_x: (torch.Tensor) 车辆之间碰撞鲁棒性在x方向的最小值，维度为 (n, H)
    min_robustness_car_car_y: (torch.Tensor) 车辆之间碰撞鲁棒性在y方向的最小值，维度为 (n, H)
    min_robustness_car_others_x: (torch.Tensor) 车辆与其他交通参与者之间碰撞鲁棒性在x方向的最小值，维度为 (n, H)
    min_robustness_car_others_y: (torch.Tensor) 车辆与其他交通参与者之间碰撞鲁棒性在y方向的最小值，维度为 (n, H)
    """

    # 提取数据
    car_indices = torch.where(data['agent']['category'] == 1)[0]
    others_indices = torch.where(data['agent']['category'] != 1)[0]

    car_positions = trajectory_proposed  # 维度 (num_cars, H, F, 2)
    others_positions = raw [others_indices ] # 维度 (num_others, H, F, 2)
    car_lengths = data['agent']['length'][car_indices]
    car_widths = data['agent']['width'][car_indices]

    # 初始化变量
    num_cars = len(car_indices)
    num_others = len(others_indices)
    H = car_positions.shape[1]  # 时间步长

    # 初始化结果数组
    min_robustness_car_car_x = torch.full((num_cars, H), 1000.0, device='cuda')
    min_robustness_car_car_y = torch.full((num_cars, H), 1000.0, device='cuda')
    min_robustness_car_others_x = torch.full((num_cars, H), 1000.0, device='cuda')
    min_robustness_car_others_y = torch.full((num_cars, H), 1000.0, device='cuda')

    # 计算car与car的鲁棒性
    for i in range(num_cars):
        traj_in = car_positions[i]
        size_in = torch.tensor([car_lengths[i], car_widths[i]], dtype=torch.float32, device='cuda')

        for t in range(H):
            min_r_ox_car = torch.tensor(1000.0, device='cuda')
            min_r_oy_car = torch.tensor(1000.0, device='cuda')

            for j in range(num_cars):
                if i != j:
                    traj_oc_in = car_positions[j]
                    size_oc_in = torch.tensor([car_lengths[j], car_widths[j]], dtype=torch.float32 ,device='cuda')

                    # 计算碰撞鲁棒性
                    r_ox, r_oy = calculate_robustness_car(traj_in[t, :, :], size_in, traj_oc_in[t, :, :], size_oc_in)
                    if r_ox < min_r_ox_car and r_ox > 0:
                        min_r_ox_car = r_ox
                    if r_oy < min_r_oy_car and r_oy > 0:
                        min_r_oy_car = r_oy

            min_robustness_car_car_x[i, t] = min_r_ox_car
            min_robustness_car_car_y[i, t] = min_r_oy_car

    # 计算car与其他交通参与者的鲁棒性
    for car_idx in range(num_cars):
        traj_in = car_positions[car_idx]

        for t in range(H):
            min_r_ox_others = torch.tensor(1000.0, device='cuda')
            min_r_oy_others = torch.tensor(1000.0, device='cuda')

            for other_idx in range(num_others):
                traj_oc_in = others_positions[other_idx]

                # 计算碰撞鲁棒性
                r_ox, r_oy = calculate_robustness_d(traj_in[t, :, :],  traj_oc_in[t, :, :])
                if r_ox < min_r_ox_others and r_ox > 0:
                    min_r_ox_others = r_ox
                if r_oy < min_r_oy_others and r_oy > 0:
                    min_r_oy_others = r_oy

            min_robustness_car_others_x[car_idx, t] = min_r_ox_others
            min_robustness_car_others_y[car_idx, t] = min_r_oy_others

    return min_robustness_car_car_x, min_robustness_car_car_y, min_robustness_car_others_x, min_robustness_car_others_y

def calculate_robustness_d(traj_in, traj_oc_in):
    traj_in = make_torch_tensor(traj_in, keep_1dim=False)
    traj_oc_in = make_torch_tensor(traj_oc_in, keep_1dim=False)

    r_ox = torch.abs(traj_in[:, 0] - traj_oc_in[:, 0])
    r_oy = torch.abs(traj_in[:, 1] - traj_oc_in[:, 1])
    return torch.min(r_ox), torch.min(r_oy)

def calculate_robustness_car(traj_in, size_in, traj_oc_in, size_oc_in):
    traj_in = make_torch_tensor(traj_in, keep_1dim=False)
    size_in = make_torch_tensor(size_in, keep_1dim=True)
    traj_oc_in = make_torch_tensor(traj_oc_in, keep_1dim=False)
    size_oc_in = make_torch_tensor(size_oc_in, keep_1dim=True)

    r_ox = torch.abs(traj_in[:, 0] - traj_oc_in[:, 0]) - size_in[0] / 2 - size_oc_in[0] / 2
    r_oy = torch.abs(traj_in[:, 1] - traj_oc_in[:, 1]) - size_in[1] / 2 - size_oc_in[1] / 2
    return torch.min(r_ox), torch.min(r_oy)

def compute_time_headway(velocities, time_gap=2):
    num_cars, num_time_steps, num_future_steps, position_dim = velocities.shape

    velocities[torch.abs(velocities) > 1000] = 0  # 处理异常速度值

    speed_magnitudes = torch.norm(velocities, dim=3)  # 维度变为 (num_cars, H, F-1)

    time_headway = torch.zeros((num_cars, num_time_steps), dtype=torch.float32, device='cuda') # 创建在相同设备上的张量
    for car_idx in range(num_cars):
        for t in range(num_time_steps):
            valid_speeds = speed_magnitudes[car_idx, t]  # 当前时间步的速度
            valid_speeds = torch.abs(valid_speeds)  # 过滤掉速度为0的情况
            if valid_speeds.numel() > 0:
                mean_time_headway = torch.mean(valid_speeds) * time_gap  # 计算平均车头时距
            else:
                mean_time_headway = 0  # 若没有有效速度
            time_headway[car_idx, t] = mean_time_headway

    return time_headway


def compute_robustness_speed_from_positions(trajectory_proposed):
    num_cars, num_time_steps, num_future_steps, position_dim = trajectory_proposed.shape

    velocities = (trajectory_proposed[:, 1:] - trajectory_proposed[:, :-1]) * 10  # 维度变为 (num_cars, H-1, F, 2)
    velocities[torch.abs(velocities) > 1000] = 0  # 处理异常速度值

    speed_magnitudes = torch.norm(velocities, dim=3)  # 维度变为 (num_cars, H-1, F)

    r_speed_all = torch.zeros((num_cars,), dtype=torch.float32)

    for car_idx in range(num_cars):
        r_speed_array = torch.max(speed_magnitudes[car_idx], dim=1)[0]  # 取每个时间步的最大速度
        r_speed_all[car_idx] = torch.max(r_speed_array)  # 计算鲁棒性

    return r_speed_all

def robustness(data, raw, trajectory_proposed):
    car_velocites_sum, target_mask = generate_target(position=data['agent']['velocity'],
                                                     mask=data['agent']['visible_mask'],
                                                     num_historical_steps=10,
                                                     num_future_steps=30)
    car_indices = torch.where(data['agent']['category'] == 1)[0]
    others_indices = torch.where(data['agent']['category'] != 1)[0]

    car_positions = trajectory_proposed
    others_positions = raw[others_indices]

    car_velocities = car_velocites_sum[car_indices]
    others_velocities = car_velocites_sum[others_indices]

    min_robustness_car_car_x, min_robustness_car_car_y, min_robustness_car_others_x, min_robustness_car_others_y = compute_robustness_collision(
        raw, trajectory_proposed, data)

    velocities = (car_positions[:, :, 1:] - car_positions[:, :, :-1]) * 10  # 维度变为 (num_cars, H, F-1, 2)
    time_headway = compute_time_headway(velocities, time_gap=2)  # 维度变为 (num_cars, H-1)

    min_other_ttc, min_vehicle_ttc = calculate_ettc(car_positions, car_velocities, others_positions, others_velocities)
    delta_other_ttc = min_other_ttc - 1.6  # (num_cars, H)
    delta_vehicle_ttc = min_vehicle_ttc - 1.5  # (num_cars, H)
    delta_min_car_car_x = torch.min(min_robustness_car_car_x, min_robustness_car_car_y) - 2  # (num_cars, H)
    delta_max_car_car_y = torch.max(min_robustness_car_car_x, min_robustness_car_car_y) - time_headway  # (num_cars, H)
    delta_min_car_others_x = torch.min(min_robustness_car_others_x, min_robustness_car_others_y) - 1.5  # (num_cars, H)
    delta_max_car_others_x = torch.max(min_robustness_car_others_x, min_robustness_car_others_y) - time_headway  # (num_cars, H)

    deltas = {
        'delta_other_ttc': delta_other_ttc,
        'delta_vehicle_ttc': delta_vehicle_ttc,
        'delta_min_car_car_x': delta_min_car_car_x,
        'delta_max_car_car_y': delta_max_car_car_y,
        'delta_min_car_others_x': delta_min_car_others_x,
        'delta_max_car_others_x': delta_max_car_others_x
    }

    threshold_dict = {
        'delta_other_ttc': 1.6,
        'delta_vehicle_ttc': 1.5,
        'delta_min_car_car_x': 2,
        'delta_max_car_car_y': torch.mean(time_headway).item(),
        'delta_min_car_others_x': 1.5,
        'delta_max_car_others_x': torch.mean(time_headway).item()
    }

    results = torch.zeros_like(delta_other_ttc, device='cuda')

    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            car_deltas = {key: value[i, j] for key, value in deltas.items()}

            negative_values = {key: value for key, value in car_deltas.items() if (value < 0).any()}

            if len(negative_values) == 1:
                results[i, j] = list(negative_values.values())[0].min()
            elif len(negative_values) > 1:
                percentage_above_thresholds = {}
                for key, value in negative_values.items():
                    threshold = threshold_dict[key]
                    percentage = (torch.abs(value) / threshold) * 100
                    percentage_above_thresholds[key] = percentage

                max_percentage_key = max(percentage_above_thresholds, key=percentage_above_thresholds.get)
                results[i, j] = negative_values[max_percentage_key].min()
            else:
                results[i, j] = 0

    return results


    # print("Results:", results)
    #  results = np.zeros((12, 1))

    # # 对每辆车进行逐元素处理
    #  for i in range(12):
    #     car_deltas = {key: value[i, :] for key, value in deltas.items()}

    #     negative_values = {key: value for key, value in car_deltas.items() if np.any(value < 0)}

    #     if len(negative_values) == 1:
    #         # 如果只有一个小于0的值，直接输出这个值
    #         results[i] = np.min(list(negative_values.values())[0])
    #     elif len(negative_values) > 1:
    #         # 计算每个负值超出阈值的百分比
    #         percentage_above_thresholds = {}
    #         for key, value in negative_values.items():
    #             threshold = threshold_dict[key]
    #             percentage = (abs(np.min(value)) / threshold) * 100
    #             percentage_above_thresholds[key] = percentage

    #         # 选择百分比最大的负值
    #         max_percentage_key = max(percentage_above_thresholds, key=percentage_above_thresholds.get)
    #         results[i] = np.min(negative_values[max_percentage_key])
    #     else:
    #         results[i] = 0


