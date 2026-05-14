import torch

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


def _sample_one_per_row(mask: torch.Tensor, device: torch.device):
    n, _ = mask.shape
    out = torch.full((n,), -1, dtype=torch.long, device=device)
    row_has = mask.any(dim=1)
    if row_has.any():
        probs = mask[row_has].float()
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        out[row_has] = sampled
    return out


def random_allocation(
        servers,
        users,
        connect,
        fd_redundancy_k: int,
        debug_print: bool = False,
):
    device = users.device
    B, U, _ = users.shape
    S = servers.shape[1]
    k = max(1, int(fd_redundancy_k))
    needs = users[:, :, 2:6]
    cap = servers[:, :, 3:7].clone()
    replica_alloc = torch.zeros((B, U, S), dtype=torch.bool, device=device)
    user_order = torch.arange(U, device=device).unsqueeze(0).expand(B, U)
    b_idx = torch.arange(B, device=device)
    for step in range(U):
        u_idx = user_order[:, step]
        need_u = needs[b_idx, u_idx]
        can_resource_rep = (cap >= need_u.unsqueeze(1)).all(dim=-1)
        can_connect = connect[b_idx, u_idx, :].bool()
        eligible_rep = can_connect & can_resource_rep
        eligible_cnt = eligible_rep.sum(dim=1)
        can_allocate = eligible_cnt >= k
        if not can_allocate.any():
            continue
        cb = b_idx[can_allocate]
        cu = u_idx[can_allocate]
        for i, b in enumerate(cb):
            mask_s = eligible_rep[b]
            probs = mask_s.float()
            probs = probs / probs.sum().clamp_min(1e-8)
            picked_s = torch.multinomial(probs, num_samples=k, replacement=False)
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
    return alloc_num, alloc_ratio, fd_satisfy_ratio
