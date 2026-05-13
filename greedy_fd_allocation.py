import torch
from torch.utils.data import DataLoader

from dataset.user_gen_online import gen_dataset


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


def _argmax_one_per_row(mask: torch.Tensor, score: torch.Tensor, device: torch.device):
    out = torch.full((mask.shape[0],), -1, dtype=torch.long, device=device)
    row_has = mask.any(dim=1)
    if row_has.any():
        masked_score = score[row_has].masked_fill(~mask[row_has], -1e9)
        out[row_has] = masked_score.argmax(dim=1)
    return out


def greedy_fd_allocation(
        servers,
        users,
        connect,
        fd_redundancy_k: int,
        debug_print: bool = False,
):
    """FD 惩罚贪心：优先不同 FD，其次剩余容量大。"""
    device = users.device
    B, U, _ = users.shape
    S = servers.shape[1]
    k = max(1, int(fd_redundancy_k))

    needs = users[:, :, 2:6]  # [B,U,4]
    cap = servers[:, :, 3:7].clone()  # [B,S,4]
    fd1 = servers[..., -1].long()  # [B,S]

    replica_alloc = torch.zeros((B, U, S), dtype=torch.bool, device=device)

    # 固定用户顺序，保证可复现
    user_order = torch.arange(U, device=device).unsqueeze(0).expand(B, U)
    b_idx = torch.arange(B, device=device)

    # 服务器 FD 等值矩阵：fd_eq_ss[b,t,s] = (fd1[b,t] == fd1[b,s])
    fd_eq_ss = fd1.unsqueeze(2) == fd1.unsqueeze(1)  # [B,S,S]

    for step in range(U):
        u_idx = user_order[:, step]  # [B]
        need_u = needs[b_idx, u_idx]  # [B,4]

        # 先判断可连接且资源满足的服务器是否至少有 K 个
        can_connect = connect[b_idx, u_idx, :].bool()  # [B,S]
        can_resource_rep = (cap >= need_u.unsqueeze(1)).all(dim=-1)
        eligible_rep = can_connect & can_resource_rep

        eligible_cnt = eligible_rep.sum(dim=1)
        can_allocate = eligible_cnt >= k
        if not can_allocate.any():
            continue

        cb = b_idx[can_allocate]
        cu = u_idx[can_allocate]

        for i, b in enumerate(cb):
            mask_s = eligible_rep[b].clone()
            chosen_local = torch.zeros(S, dtype=torch.bool, device=device)
            picked_list = []

            for _ in range(k):
                # same_fd_rep[s]=True 表示候选 s 与当前用户已选副本存在同 FD
                same_fd_rep = torch.einsum("t,ts->s", chosen_local.float(), fd_eq_ss[b].float()) > 0
                is_new_fd = (~same_fd_rep).float()  # 新 FD 优先

                remaining = cap[b].sum(dim=-1)
                sort_key = is_new_fd * 1e6 + remaining  # 先新FD，再容量
                masked_key = sort_key.masked_fill(~mask_s, -1e9)
                s = masked_key.argmax().item()

                picked_list.append(s)
                chosen_local[s] = True
                mask_s[s] = False

            picked_s = torch.tensor(picked_list, dtype=torch.long, device=device)
            replica_alloc[b, cu[i], picked_s] = True
            cap[b, picked_s] -= need_u[b]

    user_assigned = replica_alloc.sum(dim=-1) == k
    alloc_num = user_assigned.sum(dim=1).float()
    alloc_ratio = alloc_num / float(U)

    fd_satisfy_ratio = _fd_metrics(
        servers=servers,
        replica_alloc=replica_alloc,
        fd_redundancy_k=k,
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

        print("[GreedyFDPenalty][Check] assigned_user_num:", assigned_user_num.detach().cpu().tolist())
        print("[GreedyFDPenalty][Check] assigned_allocated_resource:", assigned_alloc_resource.detach().cpu().tolist())
        print("[GreedyFDPenalty][Check] allocated_server_resource:", allocated_server_resource.detach().cpu().tolist())
        print("[GreedyFDPenalty][Check] k_ok_allocated_resource:", k_ok_alloc_resource.detach().cpu().tolist())
        print("[GreedyFDPenalty][Check] k_not_ok_allocated_resource:", k_not_ok_alloc_resource.detach().cpu().tolist())

    # 资源利用率相关指标暂不纳入返回
    return alloc_num, alloc_ratio, fd_satisfy_ratio


if __name__ == "__main__":
    train_data_size = {"test": 10}
    miu = 80
    sigma = 20
    radius_low = 3
    radius_high = 7

    batch_size = 2
    user_num = 3500
    server_percent = 50

    dataset_save_path = "../dataset/telecom/data_test"
    server_path = "../dataset/dataset-telecom/data_6.1~6.30_.xlsx"

    data_set = "telecom"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_type = {"test": []}
    dataset = gen_dataset(
        user_num,
        train_data_size,
        server_path,
        dataset_save_path,
        server_percent,
        radius_low,
        radius_high,
        miu,
        sigma,
        device,
        data_type,
        data_set,
    )

    test_loader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False)
    srv, usr, conn = next(iter(test_loader))
    srv, usr, conn = srv.to(device), usr.to(device), conn.to(device)

    out = greedy_fd_allocation(
        servers=srv,
        users=usr,
        connect=conn,
        fd_redundancy_k=2,
        debug_print=True,
    )

    print("[GreedyFDPenalty] output shapes:", [x.shape if torch.is_tensor(x) else type(x) for x in out])
    print("[GreedyFDPenalty] alloc_ratio:", out[1].detach().cpu().tolist())
    print("[GreedyFDPenalty] fd_satisfy_ratio:", out[2].detach().cpu().tolist())
