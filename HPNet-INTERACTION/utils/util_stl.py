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
    num_cars, H, F, _ = car_positions.shape

    vehicle_ttc = torch.full((num_cars, num_cars, H, F), 1000.0, device=car_positions.device)

    delta_pos = car_positions.unsqueeze(1) - car_positions.unsqueeze(0)
    delta_vel = car_velocities.unsqueeze(1) - car_velocities.unsqueeze(0)

    relative_speed_sq = torch.sum(delta_vel ** 2, dim=-1)
    valid_mask = relative_speed_sq != 0
    ttc = -torch.sum(delta_pos * delta_vel, dim=-1) / relative_speed_sq
    ttc[~valid_mask] = float('inf')
    ttc[ttc <= 0] = float('inf')

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
        sum_min = process_ettc_ttcx_ttcy_ttcyy(ettc.unsqueeze(1), ttcx.unsqueeze(1), ttcy.unsqueeze(1), ttcyy.unsqueeze(1))

    vehicle_ttc = calculate_vehicle_ttc(car_positions, car_velocities)
    min_vehicle_ttc = torch.min(vehicle_ttc, dim=1)[0].min(dim=-1)[0]

    return sum_min, min_vehicle_ttc



def calculate_robustness_d(traj_in, traj_oc_in):
    r_ox = torch.abs(traj_in[:, 0] - traj_oc_in[:, 0])
    r_oy = torch.abs(traj_in[:, 1] - traj_oc_in[:, 1])
    return torch.min(r_ox), torch.min(r_oy)

def calculate_robustness_car(traj_in, size_in, traj_oc_in, size_oc_in):
    r_ox = torch.abs(traj_in[:, 0] - traj_oc_in[:, 0]) - size_in[0] / 2 - size_oc_in[0] / 2
    r_oy = torch.abs(traj_in[:, 1] - traj_oc_in[:, 1]) - size_in[1] / 2 - size_oc_in[1] / 2
    return torch.min(r_ox), torch.min(r_oy)

def compute_robustness_collision(raw, trajectory_proposed, data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    car_indices = torch.where(data['agent']['category'] == 1)[0]
    others_indices = torch.where(data['agent']['category'] != 1)[0]

    car_positions = trajectory_proposed.to(device)
    others_positions = raw[others_indices].to(device)
    car_lengths = data['agent']['length'][car_indices].to(device)
    car_widths = data['agent']['width'][car_indices].to(device)

    num_cars = len(car_indices)
    num_others = len(others_indices)
    H = car_positions.shape[1]

    sizes_in = torch.stack([car_lengths, car_widths], dim=1).to(device)

    # Initialize robustness tensors
    min_robustness_car_car_x = torch.full((num_cars, H), 1000.0, device=device)
    min_robustness_car_car_y = torch.full((num_cars, H), 1000.0, device=device)
    min_robustness_car_others_x = torch.full((num_cars, H), 1000.0, device=device)
    min_robustness_car_others_y = torch.full((num_cars, H), 1000.0, device=device)

    for t in range(H):
        traj_in_t = car_positions[:, t, :, :]

        # Calculate car-car robustness
        for i in range(num_cars):
            traj_i_t = traj_in_t[i]
            size_in = sizes_in[i]

            r_ox_cars, r_oy_cars = [], []
            for j in range(num_cars):
                if i != j:
                    traj_j_t = traj_in_t[j]
                    size_oc_in = sizes_in[j]

                    r_ox, r_oy = calculate_robustness_car(traj_i_t, size_in, traj_j_t, size_oc_in)
                    r_ox_cars.append(r_ox)
                    r_oy_cars.append(r_oy)

            if r_ox_cars:
                min_robustness_car_car_x[i, t] = torch.min(torch.stack(r_ox_cars))
            if r_oy_cars:
                min_robustness_car_car_y[i, t] = torch.min(torch.stack(r_oy_cars))

        # Calculate car-others robustness
        for car_idx in range(num_cars):
            traj_in_t = car_positions[car_idx, t, :, :]

            r_ox_others, r_oy_others = [], []
            for other_idx in range(num_others):
                traj_oc_in = others_positions[other_idx]
                traj_oc_in_t = traj_oc_in[t, :, :]

                r_ox, r_oy = calculate_robustness_d(traj_in_t, traj_oc_in_t)
                r_ox_others.append(r_ox)
                r_oy_others.append(r_oy)

            if r_ox_others:
                min_robustness_car_others_x[car_idx, t] = torch.min(torch.stack(r_ox_others))
            if r_oy_others:
                min_robustness_car_others_y[car_idx, t] = torch.min(torch.stack(r_oy_others))

    return min_robustness_car_car_x, min_robustness_car_car_y, min_robustness_car_others_x, min_robustness_car_others_y



def compute_time_headway(velocities, time_gap=2):
    velocities[torch.abs(velocities) > 1000] = 0

    speed_magnitudes = torch.norm(velocities, dim=3)
    time_headway = torch.zeros((velocities.shape[0], velocities.shape[1]), dtype=torch.float32, device='cuda')

    mean_time_headway = torch.mean(speed_magnitudes, dim=-1) * time_gap
    time_headway[:, :] = mean_time_headway

    return time_headway




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

    # Compute collision robustness between vehicles and between vehicles and other road users
    min_robustness_car_car_x, min_robustness_car_car_y, min_robustness_car_others_x, min_robustness_car_others_y = compute_robustness_collision(
        raw, trajectory_proposed, data
    )

    # Compute velocities and time headway
    velocities = (car_positions[:, :, 1:] - car_positions[:, :, :-1]) * 10
    time_headway = compute_time_headway(velocities, time_gap=2)

    # Compute ETTC
    min_other_ttc, min_vehicle_ttc = calculate_ettc(
        car_positions, car_velocities, others_positions, others_velocities
    )

    # Calculate deltas
    delta_other_ttc = min_other_ttc - 1.6
    delta_vehicle_ttc = min_vehicle_ttc - 1.5
    delta_min_car_car_x = torch.min(min_robustness_car_car_x, min_robustness_car_car_y) - 2
    delta_max_car_car_y = torch.max(min_robustness_car_car_x, min_robustness_car_car_y) - time_headway
    delta_min_car_others_x = torch.min(min_robustness_car_others_x, min_robustness_car_others_y) - 1.5
    delta_max_car_others_x = torch.max(min_robustness_car_others_x, min_robustness_car_others_y) - time_headway

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

            if negative_values:
                percentage_above_thresholds = {key: (torch.abs(value) / threshold_dict[key]) * 100 for key, value in negative_values.items()}
                max_percentage_key = max(percentage_above_thresholds, key=percentage_above_thresholds.get)
                results[i, j] = negative_values[max_percentage_key].min()
            else:
                results[i, j] = 0

    return results


