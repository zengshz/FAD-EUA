import torch
from torch.utils.data import DataLoader
from dataset.user_gen_online import gen_dataset


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


def _argmax_one_per_row(mask: torch.Tensor, score: torch.Tensor, device: torch.device):
    out = torch.full((mask.shape[0],), -1, dtype=torch.long, device=device)
    row_has = mask.any(dim=1)
    if row_has.any():
        masked_score = score[row_has].masked_fill(~mask[row_has], -1e9)
        out[row_has] = masked_score.argmax(dim=1)
    return out


def mcf_fd_allocation(
        servers,
        users,
        connect,
        fd_redundancy_k: int,
        debug_print: bool = False,
):
    device = users.device
    B, U, _ = users.shape
    S = servers.size(1)
    k = max(1, int(fd_redundancy_k))
    needs = users[:, :, 2:6]
    cap = servers[:, :, 3:7].clone()
    fd1 = servers[..., -1].long()
    replica_alloc = torch.zeros((B, U, S), dtype=torch.bool, device=device)
    server_activated = torch.zeros(B, S, dtype=torch.bool, device=device)
    total_need = needs.sum(dim=2)
    user_order = torch.argsort(total_need, dim=1)
    b_idx = torch.arange(B, device=device)
    fd_eq_ss = fd1.unsqueeze(2) == fd1.unsqueeze(1)
    for step in range(U):
        u_idx = user_order[:, step]
        need_u = needs[b_idx, u_idx]
        can_connect = connect[b_idx, u_idx, :].bool()
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
    user_assigned = replica_alloc.sum(dim=-1) == k
    alloc_num = user_assigned.sum(dim=1).float()
    alloc_ratio = alloc_num / float(U)
    fd_satisfy_ratio = _fd_metrics(
        servers=servers,
        replica_alloc=replica_alloc,
        fd_redundancy_k=k,
    )
    return alloc_num, alloc_ratio, fd_satisfy_ratio
