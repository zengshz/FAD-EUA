import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()

        # 初始节点嵌入
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

        # 服务器节点融合 FD 标量
        self.server_fuse = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

        # 关系型二部图卷积（两层）：same-FD / diff-FD 双关系
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
        # 初始节点嵌入
        user_h = self.user_embed(users)  # [B,U,d]
        server_h = self.server_embed(servers[..., :7])  # [B,S,d]

        # FD 标量融合到服务器节点
        fd1 = servers[..., -1].float()  # [B,S]
        fd_min = fd1.min(dim=1, keepdim=True).values
        fd_max = fd1.max(dim=1, keepdim=True).values
        fd_norm = ((fd1 - fd_min) / (fd_max - fd_min).clamp_min(1e-6)).unsqueeze(-1)  # [B,S,1]
        server_h = self.server_fuse(torch.cat([server_h, fd_norm], dim=-1))  # [B,S,d]

        # 二部图连边
        connect_mask = connect.bool()  # [B,U,S]

        # 关系边：same / diff
        fd_eq_ss = fd1.unsqueeze(2) == fd1.unsqueeze(1)  # [B,S,S]
        same_fd_us = torch.einsum("bus,bst->but", connect_mask.float(), fd_eq_ss.float()) > 0  # [B,U,S]
        same_mask = connect_mask & same_fd_us
        diff_mask = connect_mask & (~same_fd_us)

        # ----- Layer 1 -----
        user_ctx_same_1 = self._mean_agg(same_mask, server_h, "bus,bsd->bud", denom_dim=-1)
        user_ctx_diff_1 = self._mean_agg(diff_mask, server_h, "bus,bsd->bud", denom_dim=-1)
        user_ctx_1 = self.user_rel_fuse_1(torch.cat([user_ctx_same_1, user_ctx_diff_1], dim=-1))
        user_h = self.user_upd_1(torch.cat([user_h, user_ctx_1], dim=-1))

        server_ctx_1 = self._mean_agg(connect_mask, user_h, "bus,bud->bsd", denom_dim=1)
        server_h = self.server_upd_1(torch.cat([server_h, server_ctx_1], dim=-1))

        # ----- Layer 2 -----
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
        d_model: int,  # 模型隐空间维度
        fd_redundancy_k: int,  # FD 冗余目标 K
    ):
        super().__init__()  # 初始化父类
        self.d_model = d_model  # 保存隐维度
        self.fd_redundancy_k = int(fd_redundancy_k)  # 保存冗余目标并转 int

        # 动态服务器状态嵌入：cap(4)
        self.state_proj = nn.Linear(4, d_model)  # 将服务器动态状态映射到 d_model

        # ===== 新增：FD gating =====
        self.fd_bias = nn.Parameter(torch.tensor(1.0))

        # 轻量单头 pointer 打分头
        self.ptr_query_proj = nn.Linear(d_model, d_model, bias=False)  # 用户 query 投影
        self.ptr_key_proj = nn.Linear(d_model, d_model, bias=False)  # 服务器 key 投影
        self.user_state_bias = nn.Linear(3, 1, bias=False)  # 用户副本状态对 logits 的轻量偏置
        self.score_bias = nn.Parameter(torch.zeros(1))  # 打分偏置


    def forward(self, user_enc, server_enc, users, servers, connect, fd_onehot, device, policy):
        B, U, _ = users.shape  # 读取 batch 大小与用户数
        S = servers.shape[1]  # 读取服务器数
        needs = users[..., 2:6]  # 用户资源需求（cpu/ram/storage/bw），形状 [B,U,4]
        init_cap = servers[..., 3:7].clone()  # 服务器初始容量（用于负载比例），形状 [B,S,4]
        cap = init_cap  # 当前可用容量（会被更新）
        k = self.fd_redundancy_k  # FD 冗余目标 K
        replica_alloc = torch.zeros((B, U, S), dtype=torch.bool, device=device)  # 副本分配矩阵 [B,U,S]

        # 策略统计
        connect_mask = connect.bool()  # 连通性掩码 [B,U,S]
        target_k = torch.full((B, U), k, dtype=torch.long, device=device)  # 每个用户目标副本数 [B,U]
        target_k_f = target_k.float()
        needs_u2 = needs.unsqueeze(2)
        inv_sqrt_d = float(self.d_model) ** -0.5
        logp_user_action = torch.zeros(B, U, device=device)  # 服务器提案动作累计得分 [B,U]

        # 服务器 FD onehot（由上游预计算并传入，避免重复计算/映射不一致）
        server_fd_onehot = fd_onehot.float()  # [B,S,F]
        user_fd_seen = torch.zeros(
            (B, U, server_fd_onehot.size(-1)),
            dtype=torch.float,
            device=device
        )

        # user_enc 在解码阶段不变，query 投影可提前一次性计算
        q = self.ptr_query_proj(user_enc)  # [B,U,d]

        while True:  # 迭代直到所有用户满足副本目标或无法继续
            replica_count = replica_alloc.sum(dim=-1)  # 当前每个用户已分配副本数 [B,U]
            need_more = replica_count < target_k  # 是否还需要补副本 [B,U]
            if not need_more.any():  # 若所有用户都满足目标
                break  # 退出循环

            # 资源可满足性 + 连接约束
            can_fulfill = (cap.unsqueeze(1) >= needs_u2).all(dim=-1)  # 资源可行掩码 [B,U,S]
            base_eligible = connect_mask & can_fulfill & need_more.unsqueeze(-1) & (~replica_alloc)  # [B,U,S]

            if not base_eligible.any():
                break

            # ===== ✅ FD gating（极简 & 无额外显存）=====
            fd_overlap = torch.einsum(
                "buf,bsf->bus",
                user_fd_seen,  # float
                server_fd_onehot  # float
            )

            is_new_fd = (fd_overlap < 0.5).float()  # [B,U,S]

            # -----------------------------
            # 阶段A：服务器提案（每轮每服务器提1个用户）
            # -----------------------------
            cap_ratio = cap / init_cap.clamp_min(1e-6)  # 剩余容量比例 [B,S,4]
            state_emb = self.state_proj(cap_ratio)  # [B,S,d]
            server_current = server_enc + state_emb  # 融合动态状态后的服务器表示 [B,S,d]

            k_vec = self.ptr_key_proj(server_current)  # [B,S,d]
            logits = (q @ k_vec.transpose(1, 2)) * inv_sqrt_d + self.score_bias  # [B,U,S]

            # 融合 gating
            logits = logits + self.fd_bias * is_new_fd

            # ===== 用户状态 bias =====
            replica_count_f = replica_count.float()
            remain_to_k = (target_k_f - replica_count_f).clamp_min(0.0)
            is_one_left = (replica_count == (target_k - 1)).float()
            user_state = torch.stack([
                replica_count_f / target_k_f.clamp_min(1.0),
                remain_to_k / target_k_f.clamp_min(1.0),
                is_one_left,
            ], dim=-1)  # [B,U,3]

            logits = logits + self.user_state_bias(user_state)  # [B,U,S]，用户状态轻量偏置（按S广播）

            candidate_scores = logits.masked_fill(~base_eligible, -1e9)

            valid_server = base_eligible.any(dim=1)
            if not valid_server.any():
                break

            valid_bs = torch.where(valid_server)  # (b,s)
            logits_su = candidate_scores.transpose(1, 2)  # [B,S,U]
            logits_server_valid = logits_su[valid_bs[0], valid_bs[1], :]  # [N_server,U]

            # 每轮每服务器提案 1 个用户
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

            # -----------------------------
            # 阶段B：用户侧直接接收最高分提案
            # -----------------------------
            acc_b = prop_b
            acc_u = prop_u
            acc_s = prop_s

            bu_key = acc_b * U + acc_u  # [N_prop]
            unique_bu, inv = torch.unique(bu_key, sorted=False, return_inverse=True)
            n_user = unique_bu.numel()
            if n_user == 0:
                break

            valid_bu_b = torch.div(unique_bu, U, rounding_mode="floor")
            valid_bu_u = unique_bu.remainder(U)

            logits_user_accept = torch.full((n_user, S), -1e9, device=device)  # [N_user,S]
            prop_score_flat = logits[acc_b, acc_u, acc_s]  # 提案分 [N_prop]
            logits_user_accept[inv, acc_s] = prop_score_flat

            # 直接选最高分提案
            chosen_s = logits_user_accept.argmax(dim=-1)

            # 直接索引写入分配，避免构造 [B,U,S] 大掩码
            sel_b = valid_bu_b
            sel_u = valid_bu_u
            sel_s = chosen_s
            replica_alloc[sel_b, sel_u, sel_s] = True

            # 增量更新用户已覆盖 FD（避免每轮全量 einsum 统计）
            user_fd_seen[prop_b, prop_u] = torch.maximum(
                user_fd_seen[prop_b, prop_u],
                server_fd_onehot[prop_b, prop_s]
            )

            # 按 (b,s) 聚合资源消耗，避免 bmm + 大掩码
            used_flat = torch.zeros((B * S, needs.size(-1)), device=device)
            flat_bs = sel_b * S + sel_s
            used_flat.index_add_(0, flat_bs, needs[sel_b, sel_u])
            used = used_flat.view(B, S, needs.size(-1))

            cap_next = cap - used
            if torch.any(cap_next < -1e-6):
                raise RuntimeError("Capacity went negative, allocation logic violated.")
            cap = cap_next.clamp_min(0.0)

            # 与服务器提案一一对应：被选中的即为 inv 对应用户的被接纳提案
            accepted_from_server = prop_s == chosen_s[inv]
            if accepted_from_server.any():
                acc_idx = torch.where(accepted_from_server)[0]
                b_acc = prop_b[acc_idx]
                u_acc = prop_u[acc_idx]
                logp_user_action[b_acc, u_acc] += chosen_server_logp_flat[acc_idx]

        logp_user = logp_user_action  # 单路策略得分（服务器提案）
        logp_accum = logp_user.sum(dim=1)  # 汇总为每个样本的累计得分
        return logp_accum, replica_alloc, cap  # 返回累计得分、副本分配、剩余容量



class AllocatorFD1(nn.Module):
    def __init__(
        self,  # 实例本身
        d_model: int,  # 隐空间维度
        dropout: float = 0.1,  # Dropout 比例
        device: str = "cuda:0",  # 设备字符串
        fd_redundancy_k: int = 2,  # FD 冗余目标 K
        policy: str = "sample",  # 默认动作策略
    ):
        super().__init__()  # 初始化父类
        # 设备与冗余度设置
        self.device = torch.device(device)  # 构造运行设备
        self.fd_redundancy_k = int(fd_redundancy_k)  # 保存 K 并转为 int
        self.policy = policy
        # 编码器与解码器
        self.encoder = Encoder(d_model=d_model, dropout=dropout)  # 初始化编码器
        self.decoder = ResourceAllocatorDecoder(
            d_model=d_model,  # 传入隐维度
            fd_redundancy_k=fd_redundancy_k,  # 传入冗余目标
        )

    @staticmethod
    def _fd_metrics(
        servers: torch.Tensor,  # 服务器张量 [B,S,8]
        replica_alloc: torch.Tensor,  # FD故障域分配 [B,U,S]
        fd_redundancy_k: int,  # 冗余目标 K
        fd_onehot: torch.Tensor = None,  # 预计算的 FD onehot
    ):
        # 计算 FD 满足率与同FD副本占比
        if fd_onehot is None:  # 若未提供 onehot
            fd1 = servers[..., -1].long()  # 读取 FD 标签 [B,S]
            fd1_min = fd1.min(dim=1, keepdim=True).values  # 每个 batch 的最小 FD
            fd1_shift = fd1 - fd1_min  # 平移到从 0 开始

            fd_classes = int(fd1_shift.max().item()) + 1  # FD 类别数
            fd_onehot = torch.nn.functional.one_hot(fd1_shift, num_classes=fd_classes).float()  # 生成 onehot

        assign_bus = replica_alloc.float()  # 转为 float 便于乘法
        user_fd_cover = torch.einsum("bus,bsf->buf", assign_bus, fd_onehot)  # 统计用户覆盖 FD
        user_fd_cnt = (user_fd_cover > 0).sum(dim=-1)  # 每用户覆盖 FD 数

        replica_count = replica_alloc.sum(dim=-1)  # 每用户副本数

        # FD 满足率：用户必须同时满足 K 个副本 且 覆盖 K 个不同的 FD
        k = max(1, int(fd_redundancy_k))
        fd_satisfy = (replica_count >= k) & (user_fd_cnt >= k)  # [B,U]
        fd_satisfy_ratio = fd_satisfy.float().mean(dim=1)  # 按用户平均得到满足率

        return fd_satisfy_ratio  # 返回 FD 满足率

    def forward(self, servers, users, connect):
        _, U, _ = users.shape  # 读取用户数

        # 预计算 FD onehot（batch 内动态 remap 为连续 ID）
        fd1 = servers[..., -1].long()  # 读取 FD 标签 [B,S]
        fd1_min = fd1.min(dim=1, keepdim=True).values  # 每个 batch 的最小 FD
        fd1_shift = fd1 - fd1_min  # 平移到从 0 开始

        fd_mapped = torch.empty_like(fd1_shift)  # 创建映射容器
        for b in range(fd1_shift.size(0)):  # 遍历 batch
            _, inv = torch.unique(fd1_shift[b], return_inverse=True)  # 重新编号
            fd_mapped[b] = inv.view_as(fd1_shift[b])  # 写回映射结果

        fd_classes = int(fd_mapped.max().item()) + 1  # FD 类别数
        fd_onehot = torch.nn.functional.one_hot(fd_mapped, num_classes=fd_classes).float()  # 生成 onehot

        # 编码
        user_enc, server_enc = self.encoder(
            users=users,  # 用户输入
            servers=servers,  # 服务器输入
            connect=connect,  # 连通矩阵
        )

        # 解码
        logp_accum, replica_alloc, _ = self.decoder(
            user_enc=user_enc,  # 用户编码
            server_enc=server_enc,  # 服务器编码
            users=users,  # 用户原始特征
            servers=servers,  # 服务器原始特征
            connect=connect,  # 连通矩阵
            fd_onehot=fd_onehot,  # 复用预计算 FD onehot
            device=self.device,  # 设备
            policy=self.policy,  # 动作策略：sample/greedy
        )

        # 分配率：只有满足 K 个副本的用户才算分配成功
        replica_count = replica_alloc.sum(dim=-1)  # 每用户副本数
        k = max(1, int(self.fd_redundancy_k))
        alloc_satisfied = replica_count >= k  # [B,U]
        alloc_num = alloc_satisfied.sum(dim=1).float()  # 成功分配用户数
        alloc_ratio = alloc_num / float(U)  # 分配率

        # FD 指标
        fd_satisfy_ratio = self._fd_metrics(
            servers=servers,  # 服务器信息
            replica_alloc=replica_alloc,  # FD故障域分配
            fd_redundancy_k=self.fd_redundancy_k,  # 冗余目标
            fd_onehot=fd_onehot,  # FD onehot
        )

        utility = (
            0.5 * fd_satisfy_ratio
            + 0.5 * alloc_ratio
        )

        # 训练与评估辅助统计
        aux = {
            "alloc_ratio": alloc_ratio,  # 分配率
            "fd_satisfy_ratio": fd_satisfy_ratio,  # FD 满足率
        }
        return -utility, logp_accum, alloc_num, aux









