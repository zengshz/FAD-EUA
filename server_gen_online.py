
import numpy as np
import pandas as pd
import os
import math


def load_machine_meta_fd1_distribution(meta_path):
    if not os.path.exists(meta_path):
        print(f"警告: 找不到 {meta_path}，将使用均匀分布。")
        return None, None
    df = pd.read_csv(meta_path, header=None)
    fd1_counts = df[2].value_counts()
    fd1_values = fd1_counts.index.values
    fd1_probs = fd1_counts.values / fd1_counts.sum()
    return fd1_values, fd1_probs


def _sample_fd1_compact(num_servers, fd1_values, fd1_probs):
    if num_servers <= 0:
        return np.array([], dtype=np.int64)
    adaptive_fd_count = max(8, int(np.floor(2.0 * np.sqrt(num_servers))))
    target_fd_count = min(adaptive_fd_count, len(fd1_values), num_servers)
    min_per_fd = 3
    if target_fd_count * min_per_fd > num_servers:
        target_fd_count = max(1, num_servers // min_per_fd)
    if target_fd_count <= 0:
        target_fd_count = 1
    prob = np.asarray(fd1_probs, dtype=np.float64)
    prob = prob / prob.sum()
    chosen_idx = np.random.choice(len(fd1_values), size=target_fd_count, replace=False, p=prob)
    chosen_fd = np.asarray(fd1_values)[chosen_idx]
    chosen_prob = prob[chosen_idx]
    chosen_prob = chosen_prob / chosen_prob.sum()
    base = np.full(target_fd_count, min_per_fd, dtype=np.int64)
    remaining = num_servers - int(base.sum())
    if remaining < 0:
        counts = np.random.multinomial(num_servers, chosen_prob)
    else:
        counts = base + np.random.multinomial(remaining, chosen_prob)
    fd_assign = np.repeat(chosen_fd, counts)
    np.random.shuffle(fd_assign)
    return fd_assign


def miller_to_xy(lons, lats):
    L = 6381372 * math.pi * 2
    W = L
    H = L / 2
    mill = 2.3
    x_rad = np.radians(lons)
    y_rad = np.radians(lats)
    y_trans = 1.25 * np.log(np.tan(0.25 * math.pi + 0.4 * y_rad))
    x = (W / 2) + (W / (2 * math.pi)) * x_rad
    y = (H / 2) - (H / (2 * mill)) * y_trans
    return x, y


def coordinate_transformation_pipeline(coords):
    coords -= np.min(coords, axis=0)
    coords /= 100
    return coords


def gen_eua_servers_dataset(server_path, percent, radius_low, radius_high, miu, sigma, save_path):
    save_path = os.path.join(save_path, f'{percent}_miu_{miu}_sigma_{sigma}_low_{radius_low}_high_{radius_high}',
                             f'servers_pct_{percent}.csv')
    if os.path.exists(save_path):
        edge_servers = pd.read_csv(save_path)
    else:
        print("生成新服务器数据集...")
        edge_servers_list = pd.read_csv(
            server_path,
            usecols=['LONGITUDE', 'LATITUDE']
        )
        x_coords, y_coords = miller_to_xy(
            edge_servers_list['LONGITUDE'].values,
            edge_servers_list['LATITUDE'].values
        )
        coords = np.column_stack((x_coords, y_coords))
        transformed_coords = coordinate_transformation_pipeline(coords)
        total_servers = len(transformed_coords)
        sample_size = max(1, int(total_servers * percent / 100))
        filtered_idx = np.random.choice(total_servers, size=sample_size, replace=False)
        final_coords = transformed_coords[filtered_idx]
        final_coords -= np.min(final_coords, axis=0)
        edge_servers = pd.DataFrame({
            'X': final_coords[:, 0],
            'Y': final_coords[:, 1],
            'RADIUS': np.random.uniform(radius_low, radius_high, len(final_coords)),
        })
        num_servers = len(final_coords)
        resource_arr = np.random.normal(miu, sigma, size=(num_servers, 4))
        resource_arr = np.where(resource_arr < 0, 1, resource_arr)
        resource_df = pd.DataFrame(
            resource_arr,
            columns=['Resource_CPU', 'Resource_Memory', 'Resource_Storage', 'Resource_Bandwidth']
        )
        edge_servers = pd.concat([edge_servers, resource_df], axis=1)
        meta_path = os.path.join(os.path.dirname(__file__), 'machine_meta.csv')
        fd1_values, fd1_probs = load_machine_meta_fd1_distribution(meta_path)
        edge_servers['FD1'] = _sample_fd1_compact(num_servers, fd1_values, fd1_probs)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        edge_servers.to_csv(save_path, index=False)
        print('边缘服务器数据已保存到：', save_path)
    edge_servers = edge_servers.reset_index(drop=True)
    return edge_servers


def gen_telecom_servers_dataset(server_path, percent, radius_low, radius_high, miu, sigma, save_path):
    save_path = os.path.join(save_path, f'{percent}_miu_{miu}_sigma_{sigma}_low_{radius_low}_high_{radius_high}',
                             f'servers_pct_{percent}.csv')
    if os.path.exists(save_path):
        edge_servers = pd.read_csv(save_path)
    else:
        print("生成新服务器数据集...")
        df_raw = pd.read_excel(server_path)
        location_col = 'location(latitude/lontitude)'
        df_raw = (
            df_raw[location_col]
            .astype(str)
            .str.split('/', expand=True)
            .rename(columns={0: 'LATITUDE', 1: 'LONGITUDE'})
            [['LATITUDE', 'LONGITUDE']]
        )
        df_raw['LATITUDE'] = pd.to_numeric(df_raw['LATITUDE'], errors='coerce')
        df_raw['LONGITUDE'] = pd.to_numeric(df_raw['LONGITUDE'], errors='coerce')
        df_unique = df_raw.drop_duplicates(subset=['LATITUDE', 'LONGITUDE'])
        min_latitude = 31.175000
        max_latitude = 31.255000
        min_longitude = 121.390000
        max_longitude = 121.525000
        df_unique = df_unique[
            (df_unique['LATITUDE'] >= min_latitude) &
            (df_unique['LATITUDE'] <= max_latitude) &
            (df_unique['LONGITUDE'] >= min_longitude) &
            (df_unique['LONGITUDE'] <= max_longitude)
            ]
        x_coords, y_coords = miller_to_xy(
            df_unique['LONGITUDE'].values,
            df_unique['LATITUDE'].values
        )
        coords = np.column_stack((x_coords, y_coords))
        transformed_coords = coordinate_transformation_pipeline(coords)
        total_servers = len(transformed_coords)
        sample_size = max(1, int(total_servers * percent / 100))
        filtered_idx = np.random.choice(total_servers, size=sample_size, replace=False)
        final_coords = transformed_coords[filtered_idx]
        final_coords -= np.min(final_coords, axis=0)
        edge_servers = pd.DataFrame({
            'X': final_coords[:, 0],
            'Y': final_coords[:, 1],
            'RADIUS': np.random.uniform(radius_low, radius_high, len(final_coords)),
        })
        num_servers = len(final_coords)
        resource_arr = np.random.normal(miu, sigma, size=(num_servers, 4))
        resource_arr = np.where(resource_arr < 0, 1, resource_arr)
        resource_df = pd.DataFrame(
            resource_arr,
            columns=['Resource_CPU', 'Resource_Memory', 'Resource_Storage', 'Resource_Bandwidth']
        )
        edge_servers = pd.concat([edge_servers, resource_df], axis=1)
        meta_path = os.path.join(os.path.dirname(__file__), 'machine_meta.csv')
        fd1_values, fd1_probs = load_machine_meta_fd1_distribution(meta_path)
        if fd1_values is None:
            pseudo_fd_values = np.arange(max(1, num_servers))
            pseudo_probs = np.ones_like(pseudo_fd_values, dtype=np.float64)
            pseudo_probs = pseudo_probs / pseudo_probs.sum()
            edge_servers['FD1'] = _sample_fd1_compact(num_servers, pseudo_fd_values, pseudo_probs)
        else:
            edge_servers['FD1'] = _sample_fd1_compact(num_servers, fd1_values, fd1_probs)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        edge_servers.to_csv(save_path, index=False)
        print('边缘服务器数据已保存到：', save_path)
    edge_servers = edge_servers.reset_index(drop=True)
    return edge_servers



