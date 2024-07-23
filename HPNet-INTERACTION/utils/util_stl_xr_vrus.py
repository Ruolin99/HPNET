# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 22:01:52 2024

@author: Ruolin Shi
"""

import torch
import math
from typing import Tuple

#torch.set_float32_matmul_precision('high')


def generate_target(position: torch.Tensor, mask: torch.Tensor, num_historical_steps: int, num_future_steps: int) -> \
        Tuple[torch.Tensor, torch.Tensor]:
    target_traj = [position[:, i + 1:i + 1 + num_future_steps] for i in range(num_historical_steps)]
    target_traj = torch.stack(target_traj, dim=1)
    target_mask = [mask[:, i + 1:i + 1 + num_future_steps] for i in range(num_historical_steps)]
    target_mask = torch.stack(target_mask, dim=1)
    return target_traj, target_mask


def calculate_vehicle_ttc(car_positions, car_velocities):
    num_cars, H, F, _ = car_positions.shape

    vehicle_ttc = torch.full((num_cars, num_cars, H, F), 1000.0, device=car_positions.device)

    delta_pos = car_positions.unsqueeze(1) - car_positions.unsqueeze(0)
    delta_vel = car_velocities.unsqueeze(1) - car_velocities.unsqueeze(0)

    relative_speed_sq = torch.sum(delta_vel ** 2, dim=-1)
    valid_mask = relative_speed_sq != 0
    ttc = -torch.sum(delta_pos * delta_vel, dim=-1) / relative_speed_sq
    ttc[~valid_mask] = float('1000')
    ttc[ttc <= 0] = float('1000')

    vehicle_ttc = torch.min(vehicle_ttc, ttc)

    return vehicle_ttc


def process_ettc_ttcx_ttcy_ttcyy(ettc, ttcx, ttcy, ttcyy):
    min_ettc, _ = torch.min(ettc, dim=1)
    min_ttcx, _ = torch.min(ttcx, dim=1)
    min_ttcy, _ = torch.min(ttcy, dim=1)
    min_ttcyy, _ = torch.min(ttcyy, dim=1)

    min_ettc[min_ettc == float('inf')] = 1000
    min_ttcx[min_ttcx == float('inf')] = 1000
    min_ttcy[min_ttcy == float('inf')] = 1000
    min_ttcyy[min_ttcyy == float('inf')] = 1000

    min_ttc = torch.min(torch.min(min_ettc, min_ttcx), torch.min(min_ttcy, min_ttcyy))
    sum_min, _ = torch.min(min_ttc, dim=-1)

    return sum_min


def calculate_ettc(car_positions, car_velocities, others_positions, others_velocities):
    num_cars, H, F, _ = car_positions.shape
    num_others = others_positions.shape[0]

    if num_others != 0:
        delta_pos = car_positions.unsqueeze(1) - others_positions.unsqueeze(0)
        delta_vel = car_velocities.unsqueeze(1) - others_velocities.unsqueeze(0)

        relative_speed_sq = torch.sum(delta_vel ** 2, dim=-1)
        valid_mask = relative_speed_sq != 0

        ettc = torch.full((num_cars, num_others, H, F), 1000.0, device=car_positions.device)
        ttcx = torch.full((num_cars, num_others, H, F), 1000.0, device=car_positions.device)
        ttcy = torch.full((num_cars, num_others, H, F), 1000.0, device=car_positions.device)
        ttcyy = torch.full((num_cars, num_others, H, F), 1000.0, device=car_positions.device)

        ettc_temp = -torch.sum(delta_pos * delta_vel, dim=-1) / relative_speed_sq
        ettc[valid_mask] = ettc_temp[valid_mask]
        ettc[ettc <= 0] = float('inf')

        ttcx_temp = delta_pos[..., 0] / delta_vel[..., 0]
        ttcx[delta_vel[..., 0] != 0] = ttcx_temp[delta_vel[..., 0] != 0]
        ttcx[ttcx <= 0] = float('inf')

        ttcy_temp = delta_pos[..., 1] / delta_vel[..., 1]
        ttcy[delta_vel[..., 1] != 0] = ttcy_temp[delta_vel[..., 1] != 0]
        ttcy[ttcy <= 0] = float('inf')

        relative_speed = torch.norm(delta_vel, dim=-1)
        ttcyy_temp = torch.norm(delta_pos, dim=-1) / relative_speed
        ttcyy[relative_speed != 0] = ttcyy_temp[relative_speed != 0]
        ttcyy[ttcyy <= 0] = float('inf')

        sum_min = process_ettc_ttcx_ttcy_ttcyy(ettc, ttcx, ttcy, ttcyy)
    else:
        ettc = torch.full((num_cars, H, F), 1000.0, device=car_positions.device)
        ttcx = torch.full((num_cars, H, F), 1000.0, device=car_positions.device)
        ttcy = torch.full((num_cars, H, F), 1000.0, device=car_positions.device)
        ttcyy = torch.full((num_cars, H, F), 1000.0, device=car_positions.device)
        sum_min = process_ettc_ttcx_ttcy_ttcyy(ettc.unsqueeze(1), ttcx.unsqueeze(1), ttcy.unsqueeze(1),
                                               ttcyy.unsqueeze(1))

    vehicle_ttc = calculate_vehicle_ttc(car_positions, car_velocities)
    min_vehicle_ttc = torch.min(vehicle_ttc, dim=1)[0].min(dim=-1)[0]

    return sum_min, min_vehicle_ttc

def compute_min_distances(car_positions, others_positions, car_lengths, car_widths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    car_positions = car_positions.to(device)
    car_lengths = car_lengths.to(device)
    car_widths = car_widths.to(device)

    num_cars, H, F, _ = car_positions.shape
    num_others = others_positions.shape[0]

    # 初始化结果数组
    min_dist_car_to_car = torch.full((num_cars, H), float('inf'), device=device)

    min_dist_car_to_car_idx = torch.full((num_cars, H, 2), -1, device=device, dtype=torch.int)


    # 计算车辆之间的最小距离
    for i in range(num_cars):
        car_i_positions = car_positions[i].unsqueeze(0).expand(num_cars, -1, -1, -1)

        # 调整距离考虑车辆尺寸
        adjusted_dist_x = torch.abs(car_i_positions[..., 0] - car_positions[..., 0]) - (car_lengths[i] / 2 + car_lengths / 2).view(-1, 1, 1)
        adjusted_dist_y = torch.abs(car_i_positions[..., 1] - car_positions[..., 1]) - (car_widths[i] / 2 + car_widths / 2).view(-1, 1, 1)

        adjusted_distances = torch.sqrt(adjusted_dist_x ** 2 + adjusted_dist_y ** 2)

        for h in range(H):
            min_dist, idx = torch.min(adjusted_distances[:, h].reshape(-1), dim=-1)
            idx_f = idx.item() % F
            min_dist_car_to_car[i, h] = min_dist
            min_dist_car_to_car_idx[i, h] = torch.tensor([h, idx_f], device=device)

    # 计算车辆与其他交通参与者之间的最小距离


    return min_dist_car_to_car, min_dist_car_to_car_idx

def compute_headway_distances(car_speeds, car_lengths, min_dist_car_to_car_idx, time_gap):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    car_speeds = car_speeds.to(device)

    num_cars, H, F, _ = car_speeds.shape

    # 初始化结果数组
    min_headway_car_to_car = torch.zeros((num_cars, H), device=device)

    # 计算车辆之间的最小车头时距距离
    for i in range(num_cars):
        for h in range(H):
            if min_dist_car_to_car_idx[i, h, 0] != -1:
                idx_h = min_dist_car_to_car_idx[i, h, 0].item()
                idx_f = min_dist_car_to_car_idx[i, h, 1].item()
                if idx_f < F:  # 检查索引是否在范围内
                    speed_vector = car_speeds[i, idx_h, idx_f]
                    speed_magnitude = torch.sqrt(torch.sum(speed_vector ** 2))
                    min_headway_car_to_car[i, h] = speed_magnitude * time_gap-car_lengths[i]



    return min_headway_car_to_car

def robustness(data, raw, trajectory_proposed):
    # Generate target vehicle's velocity and mask
    car_velocities_sum, target_mask = generate_target(
        position=data['agent']['velocity'],
        mask=data['agent']['visible_mask'],
        num_historical_steps=10,
        num_future_steps=30
    )

    car_indices = torch.where(data['agent']['category'] == 1)[0]
    others_indices = torch.where(data['agent']['category'] != 1)[0]

    car_positions = trajectory_proposed
    others_positions = raw[others_indices]

    car_velocities = car_velocities_sum[car_indices]
    others_velocities = car_velocities_sum[others_indices]

    car_lengths = data['agent']['length'][car_indices].to('cuda')
    car_widths = data['agent']['width'][car_indices].to('cuda')
    velocities = (car_positions[:, :, 1:] - car_positions[:, :, :-1]) * 10
    # Compute collision robustness between vehicles and between vehicles and other road users
    min_dist_car_to_car, min_dist_car_to_car_idx= compute_min_distances(
        car_positions, others_positions, car_lengths, car_widths)

    # 调用计算车头时距的函数
    # Compute velocities and time headway
    velocities = (car_positions[:, :, 1:] - car_positions[:, :, :-1]) * 10
    min_headway_car_to_car = compute_headway_distances(
        velocities, car_lengths, min_dist_car_to_car_idx,time_gap=2)
    # Compute ETTC
    min_other_ttc, min_vehicle_ttc = calculate_ettc(
        car_positions, car_velocities, others_positions, others_velocities
    )

    # Calculate deltas
    delta_other_ttc = min_other_ttc - 1.6
    delta_vehicle_ttc = min_vehicle_ttc - 1.5
    delta_min_car_car_y = min_dist_car_to_car- min_headway_car_to_car


    deltas = {
        'delta_other_ttc': delta_other_ttc,
        'delta_vehicle_ttc': delta_vehicle_ttc,
        'delta_min_car_car_y': delta_min_car_car_y,

    }

    threshold_dict = {
        'delta_other_ttc': 1.6,
        'delta_vehicle_ttc': 1.5,
        'delta_min_car_car_y': torch.mean(min_headway_car_to_car).item(),

    }

    results = torch.zeros_like(delta_other_ttc, device='cuda')

    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            car_deltas = {key: value[i, j] for key, value in deltas.items()}
            negative_values = {key: value for key, value in car_deltas.items() if (value < 0).any()}

            if negative_values:
                percentage_above_thresholds = {key: (torch.abs(value) / threshold_dict[key]) * 100 for key, value in
                                               negative_values.items()}
                max_percentage_key = max(percentage_above_thresholds, key=percentage_above_thresholds.get)
                results[i, j] = negative_values[max_percentage_key].min()
            else:
                results[i, j] = 0

    return results


