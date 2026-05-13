import torch  # 导入 PyTorch 主库
from torch.utils.data import DataLoader  # 导入 DataLoader 以便批量读取测试数据
from dataset.user_gen_online import gen_dataset  # 导入数据生成/加载函数


def _fd_metrics(servers, replica_alloc, fd_redundancy_k: int):
    fd1 = servers[..., -1].long()  # [B,S]
    selected_fd = torch.where(
        replica_alloc,
        fd1.unsqueeze(1).expand_as(replica_alloc),
        torch.full_like(replica_alloc.long(), 10 ** 9),
    )  # [B,U,S]

    sorted_fd, _ = selected_fd.sort(dim=-1)
    valid = sorted_fd != 10 ** 9
    has_any = valid[..., 0].long()
    fd_new = (sorted_fd[..., 1:] != sorted_fd[..., :-1]) & valid[..., 1:]
    user_fd_cnt = has_any + fd_new.sum(dim=-1)  # [B,U]

    k = max(1, int(fd_redundancy_k))
    replica_count = replica_alloc.sum(dim=-1)  # [B,U]
    user_assigned = replica_count == k

    fd_satisfied = user_assigned & (user_fd_cnt == k)
    fd_satisfy = fd_satisfied.float().mean(dim=1)  # [B]

    return fd_satisfy


def _argmax_one_per_row(mask: torch.Tensor, score: torch.Tensor, device: torch.device):  # 定义按行取最大值索引函数
    out = torch.full((mask.shape[0],), -1, dtype=torch.long, device=device)  # 初始化输出索引，全为 -1
    row_has = mask.any(dim=1)  # 判断每行是否存在候选
    if row_has.any():  # 若存在有候选的行
        masked_score = score[row_has].masked_fill(~mask[row_has], -1e9)  # 将非法候选屏蔽为极小值
        out[row_has] = masked_score.argmax(dim=1)  # 在每行合法候选中取最大分数索引
    return out  # 返回每行选择结果


def mcf_fd_allocation(  # 定义 MCF-FD 分配主函数
        servers,  # 服务器张量，形状 [B,S,8]
        users,  # 用户张量，形状 [B,U,6]
        connect,  # 连通矩阵，形状 [B,U,S]
        fd_redundancy_k: int,  # FD 冗余目标 K
        debug_print: bool = False,  # 是否打印调试信息
):
    """MCF-FD-EUA：用户按需求升序；服务器优优先新 FD，其次已激活，最后剩余资源"""  # 算法说明
    device = users.device  # 获取当前计算设备
    B, U, _ = users.shape  # 读取 batch 大小与用户数量
    S = servers.size(1)  # 读取服务器数量
    k = max(1, int(fd_redundancy_k))  # 确保 K 至少为 1

    needs = users[:, :, 2:6]  # 提取用户资源需求，形状 [B,U,4]
    cap = servers[:, :, 3:7].clone()  # 复制服务器剩余容量，形状 [B,S,4]

    fd1 = servers[..., -1].long()  # 服务器 FD 标签 [B,S]

    replica_alloc = torch.zeros((B, U, S), dtype=torch.bool, device=device)  # 副本分配矩阵，True 表示用户分配到该服务器
    server_activated = torch.zeros(B, S, dtype=torch.bool, device=device)  # 记录服务器是否已被激活

    total_need = needs.sum(dim=2)  # 计算每个用户总需求，形状 [B,U]
    user_order = torch.argsort(total_need, dim=1)  # 按总需求升序得到用户处理顺序
    b_idx = torch.arange(B, device=device)  # 构造 batch 索引向量

    fd_eq_ss = fd1.unsqueeze(2) == fd1.unsqueeze(1)  # [B,S,S]

    for step in range(U):  # 按排序顺序逐个处理用户
        u_idx = user_order[:, step]  # 取当前轮每个 batch 的用户索引
        need_u = needs[b_idx, u_idx]  # 取当前用户资源需求，形状 [B,4]

        # 先判断可连接且资源满足的服务器是否至少有 K 个
        can_connect = connect[b_idx, u_idx, :].bool()  # 判断连通性，形状 [B,S]
        can_resource_rep = (cap >= need_u.unsqueeze(1)).all(dim=-1)
        eligible_rep = can_connect & can_resource_rep

        eligible_cnt = eligible_rep.sum(dim=1)
        can_allocate = eligible_cnt >= k
        if not can_allocate.any():
            continue

        cb = b_idx[can_allocate]
        cu = u_idx[can_allocate]

        # FD 优先：优先新 FD，其次已激活，最后剩余资源，连续选 K 台不同服务器
        for i, b in enumerate(cb):
            mask_s = eligible_rep[b].clone()
            chosen_local = torch.zeros(S, dtype=torch.bool, device=device)
            picked_list = []

            for _ in range(k):
                same_fd_rep = torch.einsum("t,ts->s", chosen_local.float(), fd_eq_ss[b].float()) > 0
                is_new_fd = (~same_fd_rep).float()
                activated = server_activated[b].float()
                remaining = cap[b].sum(dim=-1)
                sort_key = is_new_fd * 1e9 + activated * 1e6 + remaining
                masked_key = sort_key.masked_fill(~mask_s, -1e9)
                s = masked_key.argmax().item()

                picked_list.append(s)
                chosen_local[s] = True
                mask_s[s] = False

            picked_s = torch.tensor(picked_list, dtype=torch.long, device=device)
            replica_alloc[b, cu[i], picked_s] = True
            server_activated[b, picked_s] = True
            cap[b, picked_s] -= need_u[b]


    user_assigned = replica_alloc.sum(dim=-1) == k  # 仅完整分配 K 副本才算已分配 [B,U]
    alloc_num = user_assigned.sum(dim=1).float()  # 统计每个样本成功分配用户数
    alloc_ratio = alloc_num / float(U)  # 计算每个样本的用户分配比例

    fd_satisfy_ratio = _fd_metrics(  # 计算 FD 指标
        servers=servers,  # 传入服务器信息
        replica_alloc=replica_alloc,  # 传入副本分配结果
        fd_redundancy_k=k,  # 传入 K 值
    )

    if debug_print:
        total_server_resource = servers[:, :, 3:7].sum(dim=(1, 2))
        remain_server_resource = cap.sum(dim=(1, 2))
        allocated_server_resource = total_server_resource - remain_server_resource

        selected_fd = torch.where(
            replica_alloc,
            fd1.unsqueeze(1).expand_as(replica_alloc),
            torch.full_like(replica_alloc.long(), 10 ** 9),
        )
        sorted_fd, _ = selected_fd.sort(dim=-1)
        valid_fd = sorted_fd != 10 ** 9
        has_any_fd = valid_fd[..., 0].long()
        fd_new = (sorted_fd[..., 1:] != sorted_fd[..., :-1]) & valid_fd[..., 1:]
        user_fd_cnt = has_any_fd + fd_new.sum(dim=-1)
        replica_count = replica_alloc.sum(dim=-1)
        user_assigned = replica_count == k
        user_k_ok = user_assigned & (user_fd_cnt == k)
        user_k_not_ok = user_assigned & (~user_k_ok)

        assigned_user_num = user_assigned.sum(dim=1)

        need_sum_per_user = needs.sum(dim=-1)
        allocated_resource_per_user = need_sum_per_user * replica_count.float()

        assigned_alloc_resource = (allocated_resource_per_user * user_assigned.float()).sum(dim=1)
        k_ok_alloc_resource = (allocated_resource_per_user * user_k_ok.float()).sum(dim=1)
        k_not_ok_alloc_resource = (allocated_resource_per_user * user_k_not_ok.float()).sum(dim=1)

        print("[MCF-FD][Check] assigned_user_num:", assigned_user_num.detach().cpu().tolist())
        print("[MCF-FD][Check] assigned_allocated_resource:", assigned_alloc_resource.detach().cpu().tolist())
        print("[MCF-FD][Check] allocated_server_resource:", allocated_server_resource.detach().cpu().tolist())
        print("[MCF-FD][Check] k_ok_allocated_resource:", k_ok_alloc_resource.detach().cpu().tolist())
        print("[MCF-FD][Check] k_not_ok_allocated_resource:", k_not_ok_alloc_resource.detach().cpu().tolist())

    # 资源利用率相关指标暂不纳入返回
    return alloc_num, alloc_ratio, fd_satisfy_ratio


if __name__ == "__main__":  # 脚本直跑入口
    train_data_size = {"test": 10}  # 设置测试集样本数量
    miu = 80  # 数据生成参数 μ
    sigma = 20  # 数据生成参数 σ
    radius_low = 3  # 用户覆盖半径下界
    radius_high = 7  # 用户覆盖半径上界

    batch_size = 1  # 调试时的 batch 大小
    user_num = 3500  # 每个样本用户数
    server_percent = 50  # 服务器比例参数

    dataset_save_path = "../dataset/telecom/data_test"  # 数据缓存目录
    server_path = "../dataset/dataset-telecom/data_6.1~6.30_.xlsx"  # 服务器数据文件路径

    data_set = "telecom"  # 数据集名称
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 自动选择 GPU/CPU 设备

    data_type = {"test": []}  # 构造数据容器（兼容 gen_dataset 接口）
    dataset = gen_dataset(  # 调用数据生成/加载函数
        user_num,  # 传入用户数量
        train_data_size,  # 传入数据规模配置
        server_path,  # 传入服务器数据路径
        dataset_save_path,  # 传入缓存目录路径
        server_percent,  # 传入服务器比例参数
        radius_low,  # 传入半径下界
        radius_high,  # 传入半径上界
        miu,  # 传入 μ 参数
        sigma,  # 传入 σ 参数
        device,  # 传入运行设备
        data_type,  # 传入数据容器
        data_set,  # 传入数据集标识
    )

    test_loader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False)  # 构建测试集加载器
    srv, usr, conn = next(iter(test_loader))  # 读取一个 batch 样本
    srv, usr, conn = srv.to(device), usr.to(device), conn.to(device)  # 将样本移动到目标设备

    out = mcf_fd_allocation(  # 调用 MCF-FD 分配函数
        servers=srv,  # 输入服务器张量
        users=usr,  # 输入用户张量
        connect=conn,  # 输入连通矩阵
        fd_redundancy_k=2,  # 设置 FD 冗余目标 K=2
        debug_print=True,  # 开启调试输出
    )

    print("[MCFFD] output shapes:", [x.shape if torch.is_tensor(x) else type(x) for x in out])  # 打印输出形状
    print("[MCFFD] alloc_ratio:", out[1].detach().cpu().tolist())  # 打印用户分配比例
    print("[MCFFD] fd_satisfy_ratio:", out[2].detach().cpu().tolist())  # 打印 FD 满足率
