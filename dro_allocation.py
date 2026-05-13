import torch  # 导入 PyTorch 主库
from torch.utils.data import DataLoader  # 导入 DataLoader 以便批量读取测试数据
from dataset.user_gen_online import gen_dataset  # 导入数据生成/加载函数

EL, VL, L, M, H, VH, EH = 0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1  # 定义模糊等级常量
omega_dic = {  # 定义模糊规则表
    'ML': {"SL": EL, "SM": VL, "SH": VL},  # 低均值场景下，不同方差对应权重
    'MM': {"SL": M, "SM": L, "SH": VL},  # 中均值场景下，不同方差对应权重
    'MH': {"SL": EH, "SM": VH, "SH": H},  # 高均值场景下，不同方差对应权重
}  # 模糊规则表结束


def get_fuzzy_weight(mu, std):  # 根据均值与标准差选择模糊权重
    if mu <= 0.09:  # 若均值较低
        a = 'ML'  # 归为 ML 档
    elif 0.09 < mu <= 0.22:  # 若均值中等
        a = 'MM'  # 归为 MM 档
    else:  # 若均值较高
        a = 'MH'  # 归为 MH 档

    if std <= 0.03:  # 若方差较小
        b = 'SL'  # 归为 SL 档
    elif 0.03 < std <= 0.12:  # 若方差中等
        b = 'SM'  # 归为 SM 档
    else:  # 若方差较大
        b = 'SH'  # 归为 SH 档

    return omega_dic[a][b]  # 返回对应模糊权重


def _fd_metrics(servers, replica_alloc, fd_redundancy_k: int):  # 定义故障域相关评估指标函数
    fd1 = servers[..., -1].long()  # 服务器 FD 标签，形状 [B,S]
    selected_fd = torch.where(  # 取每个用户被分配到的 FD 标签
        replica_alloc,
        fd1.unsqueeze(1).expand_as(replica_alloc),
        torch.full_like(replica_alloc.long(), 10**9),  # 未分配位置填充大数哨兵
    )  # [B,U,S]

    sorted_fd, _ = selected_fd.sort(dim=-1)  # 按 FD 排序，哨兵会排到最后
    valid = sorted_fd != 10**9  # 有效 FD 标记 [B,U,S]
    has_any = valid[..., 0].long()  # 是否至少有一个副本 [B,U]
    fd_new = (sorted_fd[..., 1:] != sorted_fd[..., :-1]) & valid[..., 1:]  # 新 FD 起点 [B,U,S-1]
    user_fd_cnt = has_any + fd_new.sum(dim=-1)  # 每用户覆盖 FD 数 [B,U]

    k = max(1, int(fd_redundancy_k))
    replica_count = replica_alloc.sum(dim=-1)  # 每用户副本数 [B,U]
    user_assigned = replica_count == k  # 仅完整分配 K 副本才算已分配

    fd_satisfied = user_assigned & (user_fd_cnt == k)  # K 副本都在不同 FD
    fd_satisfy = fd_satisfied.float().mean(dim=1)  # 满足比例（分母为总用户数）

    return fd_satisfy  # 返回 FD 满足率


def dro_allocation(  # 定义 DRO-EUA 主函数
        servers,  # 服务器张量，形状 [B,S,8]
        users,  # 用户张量，形状 [B,U,6]
        connect,  # 连通矩阵，形状 [B,U,S]
        fd_redundancy_k: int,  # FD 冗余目标 K
        gamma=1.5,  # 新激活惩罚系数
        debug_print: bool = False,  # 是否打印调试信息
):
    B, U, _ = users.shape  # 读取 batch 大小和用户数
    S = servers.size(1)  # 读取服务器数
    device = users.device  # 获取运行设备
    k = max(1, int(fd_redundancy_k))  # 保证 K 至少为 1

    needs = users[:, :, 2:6]  # 提取用户资源需求，形状 [B,U,4]
    cap = servers[:, :, 3:7].clone()  # 复制服务器剩余容量，形状 [B,S,4]

    replica_alloc = torch.zeros((B, U, S), dtype=torch.bool, device=device)  # 副本分配矩阵
    server_activated = torch.zeros(B, S, dtype=torch.bool, device=device)  # 记录服务器是否已激活

    b_idx = torch.arange(B, device=device)  # batch 索引向量

    for i in range(U):  # 逐个用户处理
        need_u = needs[:, i, :]  # 当前用户需求，形状 [B,4]

        # 先判断可连接且资源满足的服务器是否至少有 K 个
        can_connect = connect[b_idx, i, :].bool()  # 当前用户连通性掩码，形状 [B,S]
        can_resource_rep = (cap >= need_u.unsqueeze(1)).all(dim=-1)  # 资源可满足掩码 [B,S]
        eligible_rep = can_connect & can_resource_rep  # 候选掩码 [B,S]

        eligible_cnt = eligible_rep.sum(dim=1)  # 每个 batch 可选服务器数量 [B]
        can_allocate = eligible_cnt >= k  # 是否满足至少 K 个候选 [B]
        if not can_allocate.any():
            continue

        cb = b_idx[can_allocate]  # 可分配样本 batch 索引

        for _, b in enumerate(cb):
            mask_s = eligible_rep[b].clone()  # 当前候选掩码 [S]
            picked_list = []

            # 计算所有服务器资源利用率（用于 mu/std 与 B）
            capacity_used = 1 - cap[b] / servers[b, :, 3:7].clamp_min(1e-6)  # [S,4]
            used_mean = capacity_used.mean(dim=-1)  # [S]
            mu = used_mean.mean()
            std = used_mean.std(unbiased=True) if used_mean.shape[0] >= 2 else torch.zeros_like(mu)

            omega = get_fuzzy_weight(mu, std)

            for _ in range(k):
                # 仅对当前候选集合计算 C/B，并做归一化
                cand_idx = torch.where(mask_s)[0]
                if cand_idx.numel() == 0:
                    break

                # C: 是否激活（未激活惩罚）
                zi = torch.where(server_activated[b, cand_idx], torch.full((cand_idx.numel(),), 10.0, device=device),
                                 torch.zeros((cand_idx.numel(),), device=device))
                c_raw = (10.0 - zi).abs()
                C = torch.where(zi < 10.0, c_raw * gamma, c_raw)  # 激活=0，未激活=15

                # B: 资源利用率
                Bv = used_mean[cand_idx]

                max_c, min_c = C.max(), C.min()
                max_b, min_b = Bv.max(), Bv.min()

                Cn = (C - min_c) / (max_c - min_c) if (max_c - min_c) != 0 else torch.zeros_like(C)
                Bn = (Bv - min_b) / (max_b - min_b) if (max_b - min_b) != 0 else torch.zeros_like(Bv)

                S_score = omega * Cn + (1 - omega) * Bn  # 越小越优
                s_pos = S_score.argmin().item()
                s = cand_idx[s_pos].item()

                picked_list.append(s)
                mask_s[s] = False  # 无放回

            if len(picked_list) == k:
                picked_s = torch.tensor(picked_list, dtype=torch.long, device=device)
                replica_alloc[b, i, picked_s] = True
                server_activated[b, picked_s] = True
                cap[b, picked_s] -= need_u[b]


    # 只有完整分配到 K 个副本的用户才计入分配率
    user_assigned = replica_alloc.sum(dim=-1) == k  # [B,U]
    alloc_num = user_assigned.sum(dim=1).float()  # 每个样本成功分配用户数
    alloc_ratio = alloc_num / float(U)  # 每个样本用户分配率

    fd_satisfy_ratio = _fd_metrics(  # 计算 FD 指标
        servers=servers,  # 传入服务器信息
        replica_alloc=replica_alloc,  # 传入副本分配
        fd_redundancy_k=k,  # 传入 K
    )

    if debug_print:  # 若开启调试打印
        total_server_resource = servers[:, :, 3:7].sum(dim=(1, 2))  # 每个样本服务器总资源
        remain_server_resource = cap.sum(dim=(1, 2))  # 每个样本剩余资源
        allocated_server_resource = total_server_resource - remain_server_resource  # 服务器分配出去的资源总量

        fd1 = servers[..., -1].long()  # 服务器 FD 标签 [B,S]
        selected_fd = torch.where(
            replica_alloc,
            fd1.unsqueeze(1).expand_as(replica_alloc),
            torch.full_like(replica_alloc.long(), 10**9),
        )  # [B,U,S]
        sorted_fd, _ = selected_fd.sort(dim=-1)  # 排序后哨兵在末尾
        valid_fd = sorted_fd != 10**9  # 有效 FD 标记 [B,U,S]
        has_any_fd = valid_fd[..., 0].long()  # 是否至少有一个副本 [B,U]
        fd_new = (sorted_fd[..., 1:] != sorted_fd[..., :-1]) & valid_fd[..., 1:]  # 新 FD 起点 [B,U,S-1]
        user_fd_cnt = has_any_fd + fd_new.sum(dim=-1)  # 每个用户覆盖FD数量 [B,U]
        user_assigned = replica_alloc.sum(dim=-1) == k  # 仅完整分配K副本的用户才算已分配 [B,U]
        user_k_ok = user_assigned & (user_fd_cnt == k)  # 满足K副本且K个FD都不同
        user_k_not_ok = user_assigned & (~user_k_ok)  # 已分配但FD不满足

        replica_count = replica_alloc.sum(dim=-1)  # 每个用户副本数量 [B,U]

        assigned_user_num = user_assigned.sum(dim=1)  # 一共分配用户数

        need_sum_per_user = needs.sum(dim=-1)  # 每个用户单副本资源需求总量（4维求和）[B,U]
        allocated_resource_per_user = need_sum_per_user * replica_count.float()  # 每个用户总分配资源（按副本计）

        assigned_alloc_resource = (allocated_resource_per_user * user_assigned.float()).sum(dim=1)  # 已分配用户资源总量
        k_ok_alloc_resource = (allocated_resource_per_user * user_k_ok.float()).sum(dim=1)  # 满足FD资源总量
        k_not_ok_alloc_resource = (allocated_resource_per_user * user_k_not_ok.float()).sum(dim=1)  # 不满足FD资源总量

        print("[DRO-EUA][Check] assigned_user_num:", assigned_user_num.detach().cpu().tolist())
        print("[DRO-EUA][Check] assigned_allocated_resource:", assigned_alloc_resource.detach().cpu().tolist())
        print("[DRO-EUA][Check] allocated_server_resource:", allocated_server_resource.detach().cpu().tolist())
        print("[DRO-EUA][Check] k_ok_allocated_resource:", k_ok_alloc_resource.detach().cpu().tolist())
        print("[DRO-EUA][Check] k_not_ok_allocated_resource:", k_not_ok_alloc_resource.detach().cpu().tolist())

    return alloc_num, alloc_ratio, fd_satisfy_ratio


if __name__ == "__main__":  # 脚本直跑入口
    train_data_size = {"test": 10}  # 测试集样本数量配置
    miu = 80  # 数据生成参数 μ
    sigma = 20  # 数据生成参数 σ
    radius_low = 3  # 覆盖半径下界
    radius_high = 7  # 覆盖半径上界

    batch_size = 2  # 调试 batch 大小
    user_num = 3500  # 每样本用户数
    server_percent = 50  # 服务器比例参数

    dataset_save_path = "../dataset/telecom/data_test"  # 数据缓存路径
    server_path = "../dataset/dataset-telecom/data_6.1~6.30_.xlsx"  # 服务器数据文件路径

    data_set = "telecom"  # 数据集名称
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 自动选择运行设备

    data_type = {"test": []}  # 数据容器（兼容 gen_dataset 接口）
    dataset = gen_dataset(  # 调用数据生成/加载
        user_num,  # 用户数量
        train_data_size,  # 数据规模配置
        server_path,  # 服务器数据路径
        dataset_save_path,  # 缓存路径
        server_percent,  # 服务器比例
        radius_low,  # 半径下界
        radius_high,  # 半径上界
        miu,  # μ
        sigma,  # σ
        device,  # 设备
        data_type,  # 数据容器
        data_set,  # 数据集标识
    )

    test_loader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False)  # 构建测试集加载器
    srv, usr, conn = next(iter(test_loader))  # 读取一个 batch
    srv, usr, conn = srv.to(device), usr.to(device), conn.to(device)  # 将数据移动到目标设备

    out = dro_allocation(  # 调用 DRO-EUA 分配
        servers=srv,  # 服务器输入
        users=usr,  # 用户输入
        connect=conn,  # 连通矩阵输入
        gamma=1.5,  # 激活惩罚系数
        fd_redundancy_k=2,  # K=2
        debug_print=True,  # 开启调试打印
    )

    print("[DRO-EUA] output shapes:", [x.shape if torch.is_tensor(x) else type(x) for x in out])  # 打印输出形状
    print("[DRO-EUA] alloc_ratio:", out[1].detach().cpu().tolist())  # 打印用户分配率
    print("[DRO-EUA] fd_satisfy_ratio:", out[2].detach().cpu().tolist())  # 打印 FD 满足率
