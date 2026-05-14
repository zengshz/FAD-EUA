import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.user_embed = nn.Sequential(
            nn.Linear(6, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.server_embed = nn.Sequential(
            nn.Linear(7, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.server_fuse = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.user_rel_fuse_1 = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.user_rel_fuse_2 = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.user_upd_1 = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.server_upd_1 = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.user_upd_2 = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.server_upd_2 = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    @staticmethod
    def _mean_agg(mask: torch.Tensor, src: torch.Tensor, equation: str, denom_dim: int):
        w = mask.float()
        out = torch.einsum(equation, w, src)
        denom = w.sum(dim=denom_dim, keepdim=False).unsqueeze(-1).clamp_min(1.0)
        return out / denom

    def forward(self, users: torch.Tensor, servers: torch.Tensor, connect: torch.Tensor):
        user_h = self.user_embed(users)
        server_h = self.server_embed(servers[..., :7])
        fd1 = servers[..., -1].float()
        fd_min = fd1.min(dim=1, keepdim=True).values
        fd_max = fd1.max(dim=1, keepdim=True).values
        fd_norm = ((fd1 - fd_min) / (fd_max - fd_min).clamp_min(1e-6)).unsqueeze(-1)
        server_h = self.server_fuse(torch.cat([server_h, fd_norm], dim=-1))
        connect_mask = connect.bool()
        fd_eq_ss = fd1.unsqueeze(2) == fd1.unsqueeze(1)
        same_fd_us = torch.einsum("bus,bst->but", connect_mask.float(), fd_eq_ss.float()) > 0
        same_mask = connect_mask & same_fd_us
        diff_mask = connect_mask & (~same_fd_us)
        user_ctx_same_1 = self._mean_agg(same_mask, server_h, "bus,bsd->bud", denom_dim=-1)
        user_ctx_diff_1 = self._mean_agg(diff_mask, server_h, "bus,bsd->bud", denom_dim=-1)
        user_ctx_1 = self.user_rel_fuse_1(torch.cat([user_ctx_same_1, user_ctx_diff_1], dim=-1))
        user_h = self.user_upd_1(torch.cat([user_h, user_ctx_1], dim=-1))
        server_ctx_1 = self._mean_agg(connect_mask, user_h, "bus,bud->bsd", denom_dim=1)
        server_h = self.server_upd_1(torch.cat([server_h, server_ctx_1], dim=-1))
        user_ctx_same_2 = self._mean_agg(same_mask, server_h, "bus,bsd->bud", denom_dim=-1)
        user_ctx_diff_2 = self._mean_agg(diff_mask, server_h, "bus,bsd->bud", denom_dim=-1)
        user_ctx_2 = self.user_rel_fuse_2(torch.cat([user_ctx_same_2, user_ctx_diff_2], dim=-1))
        user_h = self.user_upd_2(torch.cat([user_h, user_ctx_2], dim=-1))
        server_ctx_2 = self._mean_agg(connect_mask, user_h, "bus,bud->bsd", denom_dim=1)
        server_h = self.server_upd_2(torch.cat([server_h, server_ctx_2], dim=-1))
        return user_h, server_h


class ResourceAllocatorDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        fd_redundancy_k: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.fd_redundancy_k = int(fd_redundancy_k)
        self.state_proj = nn.Linear(4, d_model)
        self.fd_alpha = nn.Parameter(torch.tensor(1.0))
        self.fd_beta = nn.Parameter(torch.tensor(0.0))
        self.ptr_query_proj = nn.Linear(d_model, d_model, bias=False)
        self.ptr_key_proj = nn.Linear(d_model, d_model, bias=False)
        self.user_state_bias = nn.Linear(3, 1, bias=False)
        self.score_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_enc, server_enc, users, servers, connect, fd_onehot, device, policy):
        B, U, _ = users.shape
        S = servers.shape[1]
        needs = users[..., 2:6]
        init_cap = servers[..., 3:7].clone()
        cap = init_cap
        k = self.fd_redundancy_k
        replica_alloc = torch.zeros((B, U, S), dtype=torch.bool, device=device)
        connect_mask = connect.bool()
        target_k = torch.full((B, U), k, dtype=torch.long, device=device)
        target_k_f = target_k.float()
        needs_u2 = needs.unsqueeze(2)
        inv_sqrt_d = float(self.d_model) ** -0.5
        logp_user_action = torch.zeros(B, U, device=device)
        server_fd_onehot = fd_onehot.float()
        user_fd_seen = torch.zeros(
            (B, U, server_fd_onehot.size(-1)),
            dtype=torch.float,
            device=device
        )
        q = self.ptr_query_proj(user_enc)
        while True:
            replica_count = replica_alloc.sum(dim=-1)
            need_more = replica_count < target_k
            if not need_more.any():
                break
            can_fulfill = (cap.unsqueeze(1) >= needs_u2).all(dim=-1)
            base_eligible = connect_mask & can_fulfill & need_more.unsqueeze(-1) & (~replica_alloc)
            if not base_eligible.any():
                break
            fd_overlap = torch.einsum(
                "buf,bsf->bus",
                user_fd_seen,
                server_fd_onehot
            )
            is_new_fd = (fd_overlap < 0.5).float()
            cap_ratio = cap / init_cap.clamp_min(1e-6)
            state_emb = self.state_proj(cap_ratio)
            server_current = server_enc + state_emb
            k_vec = self.ptr_key_proj(server_current)
            logits = (q @ k_vec.transpose(1, 2)) * inv_sqrt_d + self.score_bias
            fd_score = torch.tanh(self.fd_alpha * is_new_fd + self.fd_beta)
            logits = logits + fd_score
            replica_count_f = replica_count.float()
            remain_to_k = (target_k_f - replica_count_f).clamp_min(0.0)
            is_one_left = (replica_count == (target_k - 1)).float()
            user_state = torch.stack([
                replica_count_f / target_k_f.clamp_min(1.0),
                remain_to_k / target_k_f.clamp_min(1.0),
                is_one_left,
            ], dim=-1)
            logits = logits + self.user_state_bias(user_state)
            candidate_scores = logits.masked_fill(~base_eligible, -1e9)
            valid_server = base_eligible.any(dim=1)
            if not valid_server.any():
                break
            valid_bs = torch.where(valid_server)
            logits_su = candidate_scores.transpose(1, 2)
            logits_server_valid = logits_su[valid_bs[0], valid_bs[1], :]
            k_prop = 1
            if policy == "sample":
                gumbel = -torch.log(-torch.log(torch.rand_like(logits_server_valid).clamp_min(1e-9)))
                sampled_u2 = (logits_server_valid + gumbel).topk(k=k_prop, dim=-1).indices  # [N_server,1]
            elif policy == "greedy":
                sampled_u2 = logits_server_valid.topk(k=k_prop, dim=-1).indices  # [N_server,1]
            else:
                raise ValueError(f"Unknown policy: {policy}")
            chosen_server_logits2 = logits_server_valid.gather(1, sampled_u2)  # [N_server,1]
            chosen_server_logp2 = chosen_server_logits2 - torch.logsumexp(logits_server_valid, dim=-1, keepdim=True)
            prop_b = valid_bs[0]
            prop_s = valid_bs[1]
            prop_u = sampled_u2.squeeze(1)
            chosen_server_logp_flat = chosen_server_logp2.squeeze(1)
            if prop_b.numel() == 0:
                break
            acc_b = prop_b
            acc_u = prop_u
            acc_s = prop_s
            bu_key = acc_b * U + acc_u
            unique_bu, inv = torch.unique(bu_key, sorted=False, return_inverse=True)
            n_user = unique_bu.numel()
            if n_user == 0:
                break
            valid_bu_b = torch.div(unique_bu, U, rounding_mode="floor")
            valid_bu_u = unique_bu.remainder(U)
            logits_user_accept = torch.full((n_user, S), -1e9, device=device)
            prop_score_flat = logits[acc_b, acc_u, acc_s]
            logits_user_accept[inv, acc_s] = prop_score_flat
            chosen_s = logits_user_accept.argmax(dim=-1)
            sel_b = valid_bu_b
            sel_u = valid_bu_u
            sel_s = chosen_s
            replica_alloc[sel_b, sel_u, sel_s] = True
            user_fd_seen[prop_b, prop_u] = torch.maximum(
                user_fd_seen[prop_b, prop_u],
                server_fd_onehot[prop_b, prop_s]
            )
            used_flat = torch.zeros((B * S, needs.size(-1)), device=device)
            flat_bs = sel_b * S + sel_s
            used_flat.index_add_(0, flat_bs, needs[sel_b, sel_u])
            used = used_flat.view(B, S, needs.size(-1))
            cap_next = cap - used
            if torch.any(cap_next < -1e-6):
                raise RuntimeError("Capacity went negative, allocation logic violated.")
            cap = cap_next.clamp_min(0.0)
            logp_user_action[prop_b, prop_u] += chosen_server_logp_flat
        logp_user = logp_user_action
        logp_accum = logp_user.sum(dim=1)
        return logp_accum, replica_alloc, cap


class AllocatorFD1(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        device: str = "cuda:0",
        fd_redundancy_k: int = 2,
        policy: str = "sample",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.fd_redundancy_k = int(fd_redundancy_k)
        self.policy = policy
        self.encoder = Encoder(d_model=d_model, dropout=dropout)
        self.decoder = ResourceAllocatorDecoder(
            d_model=d_model,
            fd_redundancy_k=fd_redundancy_k,
        )

    @staticmethod
    def _fd_metrics(
        servers: torch.Tensor,
        replica_alloc: torch.Tensor,
        fd_redundancy_k: int,
        fd_onehot: torch.Tensor = None,
    ):
        if fd_onehot is None:
            fd1 = servers[..., -1].long()
            fd1_min = fd1.min(dim=1, keepdim=True).values
            fd1_shift = fd1 - fd1_min

            fd_classes = int(fd1_shift.max().item()) + 1
            fd_onehot = torch.nn.functional.one_hot(fd1_shift, num_classes=fd_classes).float()
        assign_bus = replica_alloc.float()
        user_fd_cover = torch.einsum("bus,bsf->buf", assign_bus, fd_onehot)
        user_fd_cnt = (user_fd_cover > 0).sum(dim=-1)
        replica_count = replica_alloc.sum(dim=-1)
        k = max(1, int(fd_redundancy_k))
        fd_satisfy = (replica_count >= k) & (user_fd_cnt >= k)
        fd_satisfy_ratio = fd_satisfy.float().mean(dim=1)
        return fd_satisfy_ratio

    def forward(self, servers, users, connect):
        _, U, _ = users.shape
        fd1 = servers[..., -1].long()
        fd1_min = fd1.min(dim=1, keepdim=True).values
        fd1_shift = fd1 - fd1_min
        fd_mapped = torch.empty_like(fd1_shift)
        for b in range(fd1_shift.size(0)):
            _, inv = torch.unique(fd1_shift[b], return_inverse=True)
            fd_mapped[b] = inv.view_as(fd1_shift[b])
        fd_classes = int(fd_mapped.max().item()) + 1
        fd_onehot = torch.nn.functional.one_hot(fd_mapped, num_classes=fd_classes).float()
        user_enc, server_enc = self.encoder(
            users=users,
            servers=servers,
            connect=connect,
        )
        logp_accum, replica_alloc, _ = self.decoder(
            user_enc=user_enc,
            server_enc=server_enc,
            users=users,
            servers=servers,
            connect=connect,
            fd_onehot=fd_onehot,
            device=self.device,
            policy=self.policy,
        )
        replica_count = replica_alloc.sum(dim=-1)
        k = max(1, int(self.fd_redundancy_k))
        alloc_satisfied = replica_count >= k
        alloc_num = alloc_satisfied.sum(dim=1).float()
        alloc_ratio = alloc_num / float(U)
        fd_satisfy_ratio = self._fd_metrics(
            servers=servers,
            replica_alloc=replica_alloc,
            fd_redundancy_k=self.fd_redundancy_k,
            fd_onehot=fd_onehot,
        )
        utility = (
            0.5 * fd_satisfy_ratio
            + 0.5 * alloc_ratio
        )
        aux = {
            "alloc_ratio": alloc_ratio,
            "fd_satisfy_ratio": fd_satisfy_ratio,
        }
        return -utility, logp_accum, alloc_num, aux









