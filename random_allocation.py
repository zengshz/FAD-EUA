import torch  # 导入 PyTorch 主库，用于张量计算
from torch.utils.data import DataLoader  # 导入数据加载器，便于批量读取生成的数据集
from dataset.user_gen_online import gen_dataset  # 导入你现有的数据生成函数


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

    # 仅当用户被完整分配到 K 个副本时才算“已分配”
    user_assigned = replica_count == k  # [B,U]

    # 指标：满足 K 个副本且都在不同 FD 的用户占比（分母为总用户数 U）
    fd_satisfied = user_assigned & (user_fd_cnt == k)  # [B,U]
    fd_satisfy = fd_satisfied.float().mean(dim=1)  # [B]

    return fd_satisfy  # 返回 FD 满足率


def _sample_one_per_row(mask: torch.Tensor, device: torch.device):  # 定义按行随机采样工具函数
    """每行随机选一个True位置，无候选返回-1。"""  # 函数说明：行为样本，列为候选服务器
    n, _ = mask.shape  # 读取行数 n（样本数）和列数（服务器数）
    out = torch.full((n,), -1, dtype=torch.long, device=device)  # 初始化输出为 -1，表示默认无可选服务器
    row_has = mask.any(dim=1)  # 判断每一行是否至少有一个 True 候选
    if row_has.any():  # 如果至少有一行存在候选
        probs = mask[row_has].float()  # 取出有效行的候选掩码并转 float 作为概率底稿
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)  # 行内归一化为合法概率分布
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)  # 每行采样 1 个索引
        out[row_has] = sampled  # 把采样结果写回对应行
    return out  # 返回每行采样的服务器索引（无候选行为 -1）


def random_allocation(  # 定义随机分配主函数（baseline）
        servers,  # 输入服务器张量 [B,S,8]
        users,  # 输入用户张量 [B,U,6]
        connect,  # 输入连通矩阵 [B,U,S]
        fd_redundancy_k: int,  # 每用户必须分配的副本数 K（不足 K 则该用户记为未分配）
        debug_print: bool = False,  # 是否打印资源与冗余调试信息
):
    device = users.device  # 获取运行设备
    B, U, _ = users.shape  # 读取 batch 大小 B 与用户数 U
    S = servers.shape[1]  # 读取服务器数 S
    k = max(1, int(fd_redundancy_k))  # 保证 K 至少为 1

    needs = users[:, :, 2:6]  # 截取用户资源需求（cpu/ram/storage/bw），形状 [B,U,4]
    cap = servers[:, :, 3:7].clone()  # 复制服务器剩余资源容量，形状 [B,S,4]

    replica_alloc = torch.zeros((B, U, S), dtype=torch.bool, device=device)  # 副本分配记录 [B,U,S]

    user_order = torch.arange(U, device=device).unsqueeze(0).expand(B, U)  # 固定用户顺序
    b_idx = torch.arange(B, device=device)  # batch 索引 [0..B-1]

    for step in range(U):  # 逐步处理每一轮到达用户
        u_idx = user_order[:, step]  # 当前轮每个 batch 被处理的用户索引，形状 [B]
        need_u = needs[b_idx, u_idx]  # 取出当前轮每个 batch 用户需求，形状 [B,4]

        can_resource_rep = (cap >= need_u.unsqueeze(1)).all(dim=-1)  # 资源可满足掩码 [B,S]
        can_connect = connect[b_idx, u_idx, :].bool()  # 连接可达掩码 [B,S]
        eligible_rep = can_connect & can_resource_rep  # 候选掩码 [B,S]

        # 先判断候选数量是否至少为 K；不足 K 直接跳过该用户
        eligible_cnt = eligible_rep.sum(dim=1)  # [B]
        can_allocate = eligible_cnt >= k  # [B]
        if not can_allocate.any():
            continue

        # 对可分配样本，在候选服务器中随机无放回选择 K 个
        cb = b_idx[can_allocate]  # 可分配样本 batch 索引
        cu = u_idx[can_allocate]  # 可分配样本用户索引

        for i, b in enumerate(cb):
            mask_s = eligible_rep[b]  # 该样本当前用户的候选服务器掩码 [S]
            probs = mask_s.float()
            probs = probs / probs.sum().clamp_min(1e-8)
            picked_s = torch.multinomial(probs, num_samples=k, replacement=False)  # 随机选 K 台不同服务器

            replica_alloc[b, cu[i], picked_s] = True  # 写入副本分配记录
            cap[b, picked_s] -= need_u[b]  # 扣减服务器资源

    # 只有完整分配到 K 个副本的用户才计入分配率
    user_assigned = replica_alloc.sum(dim=-1) == k  # [B,U]
    alloc_num = user_assigned.sum(dim=1).float()  # 每个样本成功分配用户数 [B]
    alloc_ratio = alloc_num / float(U)  # 每个样本用户分配比例 [B]

    fd_satisfy_ratio = _fd_metrics(  # 计算 FD 指标
        servers=servers,  # 传入服务器信息
        replica_alloc=replica_alloc,  # 传入副本分配结果
        fd_redundancy_k=k,  # 传入 K
    )

    if debug_print:  # 若开启调试打印
        total_server_resource = servers[:, :, 3:7].sum(dim=(1, 2))  # 每个样本服务器总资源
        remain_server_resource = cap.sum(dim=(1, 2))  # 每个样本剩余资源
        allocated_server_resource = total_server_resource - remain_server_resource  # 服务器已使用的总资源量

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
        replica_count = replica_alloc.sum(dim=-1)  # 每个用户副本数量 [B,U]

        user_assigned = replica_count == k  # 仅完整分配K副本的用户才算已分配 [B,U]
        user_k_ok = user_assigned & (user_fd_cnt == k)  # 满足K副本且K个FD都不同
        user_k_not_ok = user_assigned & (~user_k_ok)  # 已分配但FD不满足

        assigned_user_num = user_assigned.sum(dim=1)  # 一共分配用户数

        need_sum_per_user = needs.sum(dim=-1)  # 每个用户单副本资源需求总量（4维求和）[B,U]
        allocated_resource_per_user = need_sum_per_user * replica_count.float()  # 每个用户总分配资源（按副本计）

        assigned_alloc_resource = (allocated_resource_per_user * user_assigned.float()).sum(dim=1)  # 已分配用户总资源量
        k_ok_alloc_resource = (allocated_resource_per_user * user_k_ok.float()).sum(dim=1)  # 满足FD资源量
        k_not_ok_alloc_resource = (allocated_resource_per_user * user_k_not_ok.float()).sum(dim=1)  # 不满足FD资源量

        print("[RandomBaseline][Check] assigned_user_num:", assigned_user_num.detach().cpu().tolist())
        print("[RandomBaseline][Check] assigned_allocated_resource:", assigned_alloc_resource.detach().cpu().tolist())
        print("[RandomBaseline][Check] allocated_server_resource:", allocated_server_resource.detach().cpu().tolist())
        print("[RandomBaseline][Check] k_ok_allocated_resource:", k_ok_alloc_resource.detach().cpu().tolist())
        print("[RandomBaseline][Check] k_not_ok_allocated_resource:", k_not_ok_alloc_resource.detach().cpu().tolist())

    return alloc_num, alloc_ratio, fd_satisfy_ratio


if __name__ == "__main__":  # 脚本直接运行入口
    train_data_size = {"test": 10}  # 测试集样本数量配置
    miu = 80  # 数据生成参数 μ
    sigma = 20  # 数据生成参数 σ
    radius_low = 3  # 服务器覆盖半径下界
    radius_high = 7  # 服务器覆盖半径上界

    batch_size = 2  # 批大小
    user_num = 3500  # 每个样本用户数
    server_percent = 50  # 服务器抽样比例（相对原始服务器池）

    dataset_save_path = "../dataset/telecom/data_test"  # 数据缓存目录
    server_path = "../dataset/dataset-telecom/data_6.1~6.30_.xlsx"  # 原始服务器文件路径

    data_set = "telecom"  # 数据集标识
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 自动选择运行设备

    data_type = {"test": []}  # 数据容器（与 gen_dataset 接口保持一致）
    dataset = gen_dataset(  # 调用现有函数生成/加载数据
        user_num,  # 用户数量
        train_data_size,  # 数据规模配置
        server_path,  # 服务器文件路径
        dataset_save_path,  # 缓存路径
        server_percent,  # 服务器比例
        radius_low,  # 半径下界
        radius_high,  # 半径上界
        miu,  # μ
        sigma,  # σ
        device,  # 设备
        data_type,  # 数据容器
        data_set,  # 数据集名称
    )

    test_loader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False)  # 构造测试加载器
    srv, usr, conn = next(iter(test_loader))  # 取一个 batch 调试
    srv, usr, conn = srv.to(device), usr.to(device), conn.to(device)  # 搬移到目标设备

    out = random_allocation(  # 调用随机基线分配函数
        servers=srv,  # 服务器张量
        users=usr,  # 用户张量
        connect=conn,  # 连通矩阵
        fd_redundancy_k=2,  # K=2
        debug_print=True,  # 打开调试打印
    )

    print("[RandomBaseline] output shapes:", [x.shape if torch.is_tensor(x) else type(x) for x in out])  # 打印返回项形状
    print("[RandomBaseline] alloc_ratio:", out[1].detach().cpu().tolist())  # 打印用户分配率
    print("[RandomBaseline] fd_satisfy_ratio:", out[2].detach().cpu().tolist())  # 打印 K 满足率
