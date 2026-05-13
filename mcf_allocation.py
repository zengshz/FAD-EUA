import torch  # 导入 PyTorch 主库
from torch.utils.data import DataLoader  # 导入 DataLoader 以便批量读取测试数据
from dataset.user_gen_online import gen_dataset  # 导入数据生成/加载函数


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

    fd_satisfied = user_assigned & (user_fd_cnt == k)  # K 副本均在不同 FD
    fd_satisfy = fd_satisfied.float().mean(dim=1)  # 分母为总用户数 U

    return fd_satisfy


def mcf_allocation(  # 定义 MCF-EUA 主分配函数
        servers,  # 服务器张量，形状 [B,S,8]
        users,  # 用户张量，形状 [B,U,6]
        connect,  # 连通矩阵，形状 [B,U,S]
        fd_redundancy_k: int,  # FD 冗余目标 K
        debug_print: bool = False,  # 是否输出调试信息
):
    """MCF-EUA：用户按需求升序；服务器优先已激活且剩余资源大的候选。"""  # 算法描述
    device = users.device  # 获取当前运行设备
    B, U, _ = users.shape  # 读取 batch 大小与用户数量
    S = servers.size(1)  # 读取服务器数量
    k = max(1, int(fd_redundancy_k))  # 保证 K 至少为 1

    needs = users[:, :, 2:6]  # 提取用户资源需求，形状 [B,U,4]
    cap = servers[:, :, 3:7].clone()  # 复制服务器剩余容量，形状 [B,S,4]


    replica_alloc = torch.zeros((B, U, S), dtype=torch.bool, device=device)  # 副本分配矩阵，True 表示该用户分到该服务器
    server_activated = torch.zeros(B, S, dtype=torch.bool, device=device)  # 记录服务器是否已被激活

    total_need = needs.sum(dim=2)  # 计算每个用户总需求，形状 [B,U]
    user_order = torch.argsort(total_need, dim=1)  # 按总需求升序排列用户索引
    b_idx = torch.arange(B, device=device)  # 生成 batch 索引向量

    for step in range(U):  # 逐轮处理每个排序后的用户
        u_idx = user_order[:, step]  # 取本轮每个 batch 的目标用户索引
        need_u = needs[b_idx, u_idx]  # 读取该用户需求，形状 [B,4]

        # 先判断可连接且资源满足的服务器是否至少有 K 个
        can_connect = connect[b_idx, u_idx, :].bool()  # 判断用户与服务器是否连通，形状 [B,S]
        can_resource_rep = (cap >= need_u.unsqueeze(1)).all(dim=-1)  # 资源可满足掩码 [B,S]
        eligible_rep = can_connect & can_resource_rep  # 候选掩码 [B,S]

        eligible_cnt = eligible_rep.sum(dim=1)  # 每个 batch 可选服务器数量 [B]
        can_allocate = eligible_cnt >= k  # 是否满足至少 K 个候选 [B]
        if not can_allocate.any():
            continue

        cb = b_idx[can_allocate]  # 可分配样本 batch 索引
        cu = u_idx[can_allocate]  # 可分配样本用户索引

        # 对可分配样本：优先已激活服务器，再选容量剩余最大的服务器，连续选 K 台不同服务器
        for i, b in enumerate(cb):
            mask_s = eligible_rep[b].clone()  # 当前候选掩码 [S]
            picked_list = []
            for _ in range(k):
                activated_score = server_activated[b].float()  # 已激活优先
                cap_score = cap[b].sum(dim=-1)  # 总剩余容量
                sort_key = activated_score * 1e6 + cap_score  # 先激活后容量
                masked_key = sort_key.masked_fill(~mask_s, -1e9)
                s = masked_key.argmax().item()
                picked_list.append(s)
                mask_s[s] = False  # 无放回

            picked_s = torch.tensor(picked_list, dtype=torch.long, device=device)
            replica_alloc[b, cu[i], picked_s] = True  # 写入副本分配结果
            server_activated[b, picked_s] = True  # 将选中的服务器置为已激活
            cap[b, picked_s] -= need_u[b]  # 扣减服务器剩余资源


    # 只有完整分配到 K 个副本的用户才计入分配率
    user_assigned = replica_alloc.sum(dim=-1) == k  # [B,U]
    alloc_num = user_assigned.sum(dim=1).float()  # 统计每个样本成功分配的用户数
    alloc_ratio = alloc_num / float(U)  # 计算每个样本用户分配比例

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

        print("[MCF-EUA][Check] assigned_user_num:", assigned_user_num.detach().cpu().tolist())
        print("[MCF-EUA][Check] assigned_allocated_resource:", assigned_alloc_resource.detach().cpu().tolist())
        print("[MCF-EUA][Check] allocated_server_resource:", allocated_server_resource.detach().cpu().tolist())
        print("[MCF-EUA][Check] k_ok_allocated_resource:", k_ok_alloc_resource.detach().cpu().tolist())
        print("[MCF-EUA][Check] k_not_ok_allocated_resource:", k_not_ok_alloc_resource.detach().cpu().tolist())

    return alloc_num, alloc_ratio, fd_satisfy_ratio


if __name__ == "__main__":  # 脚本直跑入口
    train_data_size = {"test": 10}  # 测试数据规模配置
    miu = 80  # 数据生成参数 μ
    sigma = 20  # 数据生成参数 σ
    radius_low = 3  # 覆盖半径下界
    radius_high = 7  # 覆盖半径上界

    batch_size = 1  # 调试 batch 大小
    user_num = 3500  # 每个样本的用户数量
    server_percent = 50  # 服务器比例参数

    dataset_save_path = "../dataset/telecom/data_test"  # 数据缓存目录
    server_path = "../dataset/dataset-telecom/data_6.1~6.30_.xlsx"  # 服务器原始数据路径

    data_set = "telecom"  # 数据集名称
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 自动选择运行设备

    data_type = {"test": []}  # 构造数据容器（兼容 gen_dataset 接口）
    dataset = gen_dataset(  # 调用数据生成/读取函数
        user_num,  # 传入用户数
        train_data_size,  # 传入数据规模配置
        server_path,  # 传入服务器路径
        dataset_save_path,  # 传入缓存路径
        server_percent,  # 传入服务器比例
        radius_low,  # 传入半径下界
        radius_high,  # 传入半径上界
        miu,  # 传入 μ
        sigma,  # 传入 σ
        device,  # 传入设备
        data_type,  # 传入数据容器
        data_set,  # 传入数据集标识
    )

    test_loader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False)  # 构造测试集 DataLoader
    srv, usr, conn = next(iter(test_loader))  # 取一个 batch 做调试
    srv, usr, conn = srv.to(device), usr.to(device), conn.to(device)  # 将数据移动到目标设备

    out = mcf_allocation(  # 调用 MCF-EUA 分配函数
        servers=srv,  # 输入服务器张量
        users=usr,  # 输入用户张量
        connect=conn,  # 输入连通矩阵
        fd_redundancy_k=2,  # 设置 FD 冗余目标 K=2
        debug_print=True,  # 开启调试打印
    )

    print("[MCF] output shapes:", [x.shape if torch.is_tensor(x) else type(x) for x in out])  # 打印输出形状信息
    print("[MCF] alloc_ratio:", out[1].detach().cpu().tolist())  # 打印用户分配率
    print("[MCF] fd_satisfy_ratio:", out[2].detach().cpu().tolist())  # 打印 FD 满足率
