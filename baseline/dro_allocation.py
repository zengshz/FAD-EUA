import torch

EL, VL, L, M, H, VH, EH = 0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1
omega_dic = {
    'ML': {"SL": EL, "SM": VL, "SH": VL},
    'MM': {"SL": M, "SM": L, "SH": VL},
    'MH': {"SL": EH, "SM": VH, "SH": H},
}


def get_fuzzy_weight(mu, std):
    if mu <= 0.09:
        a = 'ML'
    elif 0.09 < mu <= 0.22:
        a = 'MM'
    else:
        a = 'MH'
    if std <= 0.03:
        b = 'SL'
    elif 0.03 < std <= 0.12:
        b = 'SM'
    else:
        b = 'SH'
    return omega_dic[a][b]


def _fd_metrics(servers, replica_alloc, fd_redundancy_k: int):
    fd1 = servers[..., -1].long()
    selected_fd = torch.where(
        replica_alloc,
        fd1.unsqueeze(1).expand_as(replica_alloc),
        torch.full_like(replica_alloc.long(), 10**9),
    )
    sorted_fd, _ = selected_fd.sort(dim=-1)
    valid = sorted_fd != 10**9
    has_any = valid[..., 0].long()
    fd_new = (sorted_fd[..., 1:] != sorted_fd[..., :-1]) & valid[..., 1:]
    user_fd_cnt = has_any + fd_new.sum(dim=-1)
    k = max(1, int(fd_redundancy_k))
    replica_count = replica_alloc.sum(dim=-1)
    user_assigned = replica_count == k
    fd_satisfied = user_assigned & (user_fd_cnt == k)
    fd_satisfy = fd_satisfied.float().mean(dim=1)
    return fd_satisfy


def dro_allocation(
        servers,
        users,
        connect,
        fd_redundancy_k: int,
        gamma=1.5,
        debug_print: bool = False,
):
    B, U, _ = users.shape
    S = servers.size(1)
    device = users.device
    k = max(1, int(fd_redundancy_k))
    needs = users[:, :, 2:6]
    cap = servers[:, :, 3:7].clone()
    replica_alloc = torch.zeros((B, U, S), dtype=torch.bool, device=device)
    server_activated = torch.zeros(B, S, dtype=torch.bool, device=device)
    b_idx = torch.arange(B, device=device)
    for i in range(U):
        need_u = needs[:, i, :]
        can_connect = connect[b_idx, i, :].bool()
        can_resource_rep = (cap >= need_u.unsqueeze(1)).all(dim=-1)
        eligible_rep = can_connect & can_resource_rep
        eligible_cnt = eligible_rep.sum(dim=1)
        can_allocate = eligible_cnt >= k
        if not can_allocate.any():
            continue
        cb = b_idx[can_allocate]
        for _, b in enumerate(cb):
            mask_s = eligible_rep[b].clone()
            picked_list = []
            capacity_used = 1 - cap[b] / servers[b, :, 3:7].clamp_min(1e-6)
            used_mean = capacity_used.mean(dim=-1)
            mu = used_mean.mean()
            std = used_mean.std(unbiased=True) if used_mean.shape[0] >= 2 else torch.zeros_like(mu)
            omega = get_fuzzy_weight(mu, std)
            for _ in range(k):
                cand_idx = torch.where(mask_s)[0]
                if cand_idx.numel() == 0:
                    break
                zi = torch.where(server_activated[b, cand_idx], torch.full((cand_idx.numel(),), 10.0, device=device),
                                 torch.zeros((cand_idx.numel(),), device=device))
                c_raw = (10.0 - zi).abs()
                C = torch.where(zi < 10.0, c_raw * gamma, c_raw)
                Bv = used_mean[cand_idx]
                max_c, min_c = C.max(), C.min()
                max_b, min_b = Bv.max(), Bv.min()
                Cn = (C - min_c) / (max_c - min_c) if (max_c - min_c) != 0 else torch.zeros_like(C)
                Bn = (Bv - min_b) / (max_b - min_b) if (max_b - min_b) != 0 else torch.zeros_like(Bv)
                S_score = omega * Cn + (1 - omega) * Bn
                s_pos = S_score.argmin().item()
                s = cand_idx[s_pos].item()
                picked_list.append(s)
                mask_s[s] = False
            if len(picked_list) == k:
                picked_s = torch.tensor(picked_list, dtype=torch.long, device=device)
                replica_alloc[b, i, picked_s] = True
                server_activated[b, picked_s] = True
                cap[b, picked_s] -= need_u[b]
    user_assigned = replica_alloc.sum(dim=-1) == k
    alloc_num = user_assigned.sum(dim=1).float()
    alloc_ratio = alloc_num / float(U)
    fd_satisfy_ratio = _fd_metrics(
        servers=servers,
        replica_alloc=replica_alloc,
        fd_redundancy_k=k,
    )
    return alloc_num, alloc_ratio, fd_satisfy_ratio
