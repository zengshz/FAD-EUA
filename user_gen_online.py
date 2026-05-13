# 用户状态
# ram cpu gpu storage bandwidth longitude latitude
# 内存 中央处理器 图形处理器 存储 带宽 经度 纬度

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm
from dataset.server_gen_online import gen_eua_servers_dataset, gen_telecom_servers_dataset

# 随机选择用户需求（假设USER_NEEDS是已定义的需求列表）
USER_NEEDS_ARRAY = np.array([
    [1, 2, 1, 2],
    [2, 3, 3, 4],
    [5, 7, 6, 6],
])  # shape: (3, 4)  # 用户资源需求配置cpu ram storage bandwidth

class EuaDataset(Dataset):  # 继承自Dataset类，Dataset是PyTorch中用于表示数据集的一个抽象类，重写__len__和__getitem__两个方法
    def __init__(self, servers, users_list, users_connect_list, device):
        self.users_list, self.users_connect_list = users_list, users_connect_list
        # 将输入参数servers转换为浮点型张量，保存到类实例的servers属性中。指定了数据类型为float32，并将张量放在device指定的设备上。
        self.servers = torch.tensor(servers, dtype=torch.float32, device=device)
        self.device = device

    def __len__(self):
        return len(self.users_list)  # 返回users_list的长度，即数据集中的条目数量。

    def __getitem__(self, index):  # 可以通过索引访问数据集中的特定元素
        # 从users_list中取出索引为index的元素，并将其转换为float32类型的torch.tensor对象，然后存储在指定的设备上。
        users = torch.tensor(self.users_list[index], dtype=torch.float32, device=self.device)
        # 布尔类型torch.tensor对象
        user_connect = torch.tensor(self.users_connect_list[index], dtype=torch.bool, device=self.device)
        return self.servers, users, user_connect


def generate_random_users_within_servers(servers, user_num):
    """
    在服务器覆盖范围内生成用户坐标（确保用户位于所选服务器的覆盖范围内）

    参数：
        servers: DataFrame，必须包含列 [X, Y, RADIUS]
        user_num: 生成的用户数量

    返回：
        user_coords: 用户坐标数组，形状 [user_num, 2]
    """
    # 验证输入列
    assert all(col in servers.columns for col in ['X', 'Y', 'RADIUS'])

    # 随机选择每个用户对应的服务器
    server_indices = np.random.choice(len(servers), size=user_num)
    selected_servers = servers.iloc[server_indices]

    # 计算每个用户的实际最大距离（基于服务器半径和比例参数）
    server_radii = selected_servers['RADIUS'].values  # 获取服务器覆盖半径 [user_num]

    min_radius = 0.0 * server_radii  # 实际最小距离
    max_radius = 1.0 * server_radii  # 实际最大距离

    # 生成极坐标参数（向量化操作）
    angles = np.random.uniform(0, 2 * np.pi, size=user_num)   # 随机角度 [user_num]
    radii = np.random.uniform(min_radius, max_radius)  # 半径随机分布

    # 转换为笛卡尔坐标偏移量
    dx = radii * np.cos(angles)  # [user_num]
    dy = radii * np.sin(angles)  # [user_num]

    # 计算绝对坐标
    user_x = selected_servers['X'].values + dx  # [user_num]
    user_y = selected_servers['Y'].values + dy  # [user_num]

    return np.column_stack((user_x, user_y))



def generate_connect_matrix(user_x_y_list, server_data):
    """
    生成用户-服务器连接矩阵（向量化实现）
    :param user_x_y_list: 用户坐标数组，形状 [user_num, 2]
    :param server_data: 服务器数据DataFrame，含X/Y/RADIUS列
    :return: 布尔连接矩阵，形状 [user_num, server_num]
    """
    # 提取服务器坐标和半径
    server_coords = server_data[['X', 'Y']].values  # [server_num, 2]
    server_radii = server_data['RADIUS'].values  # [server_num]

    # 分离用户坐标
    user_x = user_x_y_list[:, 0]  # [user_num]
    user_y = user_x_y_list[:, 1]  # [user_num]

    # 向量化计算距离矩阵
    dx = user_x[:, np.newaxis] - server_coords[:, 0]  # [user_num, server_num]
    dy = user_y[:, np.newaxis] - server_coords[:, 1]  # [user_num, server_num]
    distances = np.hypot(dx, dy)  # 等效于 sqrt(dx² + dy²)

    # 生成连接状态矩阵
    connect_matrix = distances <= server_radii
    return connect_matrix


def gen_user_dataset(server_data, user_num, set_type):
    """
    根据服务器的覆盖范围，生成用户数据并分配给不同的服务器。
    """
    users_list = []  # 用户列表
    users_connect_list = []  # 每个用户可连接到的服务器列表

    for _ in tqdm(range(set_type)):  # 每种数据类型生成多组数据
        user_x_y_list = generate_random_users_within_servers(server_data, user_num)
        user_connect_list = generate_connect_matrix(user_x_y_list, server_data)

        # 向量化随机选择（生成user_num个随机索引）
        config_indices = np.random.randint(0, len(USER_NEEDS_ARRAY), size=user_num)
        # 批量获取需求配置（替代循环）
        need_list = USER_NEEDS_ARRAY[config_indices]  # shape: (user_num, 4)

        # need_list = np.array([random.choice(USER_NEEDS) for _ in range(user_num)])
        user = np.concatenate((user_x_y_list, need_list), axis=1)
        users_list.append(user)
        users_connect_list.append(user_connect_list)

    return {
        "users_list": users_list,
        "users_connect_list": users_connect_list,
    }


# 生成数据集
def gen_dataset(user_num, data_size, server_path, save_path, server_percent, radius_low, radius_high,
                miu, sigma, device, combined_data, data_set):
    for set_type in combined_data.keys():

        # 读取服务器数据
        if data_set == 'eua':
            server_data = gen_eua_servers_dataset(server_path, server_percent, radius_low, radius_high, miu, sigma, save_path)
            userpath = os.path.join(save_path, f'{server_percent}_miu_{miu}_sigma_{sigma}_low_{radius_low}_high_{radius_high}',
                                    f'{set_type}_users_{user_num}_size_{data_size[set_type]}.npz')
            # 检查是否有现存的用户数据集
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
            # 检查是否有现存的用户数据集
            if os.path.exists(userpath):
                print(f"正在加载{set_type}用户数据集")
                data = np.load(userpath)
            else:
                print(f"正在生成{set_type}用户数据集")
                data = gen_user_dataset(server_data, user_num, data_size[set_type])
                np.savez_compressed(userpath, **data)
                print("数据集保存至：", userpath)

        combined_data[set_type] = EuaDataset(server_data.to_numpy(), **data, device=device)  # 使用EuaDataset类创建数据集实例，并将其存储在datasets字典中对应的键下
    return combined_data  # 返回数据集

