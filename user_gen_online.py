

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm
from dataset.server_gen_online import gen_eua_servers_dataset, gen_telecom_servers_dataset

USER_NEEDS_ARRAY = np.array([
    [1, 2, 1, 2],
    [2, 3, 3, 4],
    [5, 7, 6, 6],
])

class EuaDataset(Dataset):
    def __init__(self, servers, users_list, users_connect_list, device):
        self.users_list, self.users_connect_list = users_list, users_connect_list
        self.servers = torch.tensor(servers, dtype=torch.float32, device=device)
        self.device = device
    def __len__(self):
        return len(self.users_list)
    def __getitem__(self, index):
        users = torch.tensor(self.users_list[index], dtype=torch.float32, device=self.device)
        user_connect = torch.tensor(self.users_connect_list[index], dtype=torch.bool, device=self.device)
        return self.servers, users, user_connect


def generate_random_users_within_servers(servers, user_num):
    assert all(col in servers.columns for col in ['X', 'Y', 'RADIUS'])
    server_indices = np.random.choice(len(servers), size=user_num)
    selected_servers = servers.iloc[server_indices]
    server_radii = selected_servers['RADIUS'].values
    min_radius = 0.0 * server_radii
    max_radius = 1.0 * server_radii
    angles = np.random.uniform(0, 2 * np.pi, size=user_num)
    radii = np.random.uniform(min_radius, max_radius)
    dx = radii * np.cos(angles)
    dy = radii * np.sin(angles)
    user_x = selected_servers['X'].values + dx
    user_y = selected_servers['Y'].values + dy
    return np.column_stack((user_x, user_y))


def generate_connect_matrix(user_x_y_list, server_data):
    server_coords = server_data[['X', 'Y']].values
    server_radii = server_data['RADIUS'].values
    user_x = user_x_y_list[:, 0]
    user_y = user_x_y_list[:, 1]
    dx = user_x[:, np.newaxis] - server_coords[:, 0]
    dy = user_y[:, np.newaxis] - server_coords[:, 1]
    distances = np.hypot(dx, dy)
    connect_matrix = distances <= server_radii
    return connect_matrix


def gen_user_dataset(server_data, user_num, set_type):
    users_list = []
    users_connect_list = []
    for _ in tqdm(range(set_type)):
        user_x_y_list = generate_random_users_within_servers(server_data, user_num)
        user_connect_list = generate_connect_matrix(user_x_y_list, server_data)
        config_indices = np.random.randint(0, len(USER_NEEDS_ARRAY), size=user_num)
        need_list = USER_NEEDS_ARRAY[config_indices]
        user = np.concatenate((user_x_y_list, need_list), axis=1)
        users_list.append(user)
        users_connect_list.append(user_connect_list)

    return {
        "users_list": users_list,
        "users_connect_list": users_connect_list,
    }


def gen_dataset(user_num, data_size, server_path, save_path, server_percent, radius_low, radius_high,
                miu, sigma, device, combined_data, data_set):
    for set_type in combined_data.keys():
        if data_set == 'eua':
            server_data = gen_eua_servers_dataset(server_path, server_percent, radius_low, radius_high, miu, sigma, save_path)
            userpath = os.path.join(save_path, f'{server_percent}_miu_{miu}_sigma_{sigma}_low_{radius_low}_high_{radius_high}',
                                    f'{set_type}_users_{user_num}_size_{data_size[set_type]}.npz')
            if os.path.exists(userpath):
                print(f"正在加载{set_type}用户数据集")
                data = np.load(userpath)
            else:
                print(f"正在生成{set_type}用户数据集")
                data = gen_user_dataset(server_data, user_num, data_size[set_type])
                np.savez(userpath, **data)
                print("数据集保存至：", userpath)
        else:
            server_data = gen_telecom_servers_dataset(server_path, server_percent, radius_low, radius_high, miu, sigma, save_path)
            userpath = os.path.join(save_path, f'{server_percent}_miu_{miu}_sigma_{sigma}_low_{radius_low}_high_{radius_high}',
                                    f'{set_type}_users_{user_num}_size_{data_size[set_type]}.npz')
            if os.path.exists(userpath):
                print(f"正在加载{set_type}用户数据集")
                data = np.load(userpath)
            else:
                print(f"正在生成{set_type}用户数据集")
                data = gen_user_dataset(server_data, user_num, data_size[set_type])
                np.savez_compressed(userpath, **data)
                print("数据集保存至：", userpath)

        combined_data[set_type] = EuaDataset(server_data.to_numpy(), **data, device=device)
    return combined_data

