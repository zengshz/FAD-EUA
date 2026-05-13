import importlib
import subprocess
import sys


def _ensure_package(import_name, pip_name=None):
    """若缺少依赖则自动安装。"""
    try:
        importlib.import_module(import_name)
    except ImportError:
        pkg = pip_name or import_name
        print(f"[AutoInstall] 未检测到依赖 {pkg}，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


# 先确保第三方依赖可用
_ensure_package("yaml", "PyYAML")
_ensure_package("numpy")
_ensure_package("tqdm")
_ensure_package("torch")
_ensure_package("matplotlib")
_ensure_package("pandas")
_ensure_package("openpyxl")

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from AllocatorFD1 import AllocatorFD1
from dataset.user_gen_online import gen_dataset
from Utils import check_gradients

# 基础配置
train_data_size = {"train": 5000, "valid": 1000}
miu, sigma = 80, 20
radius_low, radius_high = 3, 7
batch_size = 25
user_num = 2000
server_percent = 50

dataset_save_path = "./dataset/telecom/data_train"
server_path = "./dataset/dataset-telecom/data_6.1~6.30_.xlsx"

d_model = 64
max_epochs = 100
lr = 1.0e-4

# 训练超参
early_stop_patience = 25
min_delta = 1e-4

# 学习率退火（关闭重启，使用无重启余弦退火）
fd_redundancy_k = 2

# AllocatorFD1：保留外部调参入口（与模型构造参数对齐）

data_set = "telecom"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_loaders():
    data_type = {"train": [], "valid": []}
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
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset["valid"], batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader


def train():
    torch.autograd.set_detect_anomaly(False)
    train_loader, valid_loader = build_loaders()

    model = AllocatorFD1(
        d_model=d_model,
        device=str(device),
        fd_redundancy_k=fd_redundancy_k,
        policy="sample",  # 训练阶段显式使用采样策略
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-6
    )

    model_dir = f"./model/MACAllocatorFD1/V2/{time.strftime('%m%d%H%M')}_simple_u{user_num}_s{server_percent}_k{fd_redundancy_k}"
    os.makedirs(model_dir, exist_ok=True)

    best_val = float("inf")
    patience_cnt = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = []

        model.policy = 'sample'
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}] TRAIN")
        for srv, usr, conn in pbar:
            srv, usr, conn = srv.to(device), usr.to(device), conn.to(device)

            optimizer.zero_grad(set_to_none=True)

            actor_obj, log_prob, _, aux = model(srv, usr, conn)

            advantage = actor_obj.detach()
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-6)
            loss = (advantage * log_prob).mean()

            loss.backward()
            optimizer.step()

            # check_gradients(model)

            train_loss.append(float(actor_obj.mean().item()))
            pbar.set_postfix({
                "Obj": f"{actor_obj.mean().item():.4f}",
                "Alloc%": f"{aux['alloc_ratio'].mean().item() * 100:.2f}",
                "FDsat%": f"{aux['fd_satisfy_ratio'].mean().item() * 100:.2f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

            # 参考 MAC_train_large：及时释放中间变量
            del actor_obj, log_prob, aux, advantage, loss

        model.eval()
        model.policy = 'greedy'
        val_loss = []
        val_alloc = []
        val_fdsat = []

        with torch.no_grad():
            for srv, usr, conn in valid_loader:
                srv, usr, conn = srv.to(device), usr.to(device), conn.to(device)
                actor_obj, _, _, aux = model(srv, usr, conn)
                val_loss.append(float(actor_obj.mean().item()))
                val_alloc.append(float(aux["alloc_ratio"].mean().item()))
                val_fdsat.append(float(aux["fd_satisfy_ratio"].mean().item()))

        train_m = float(np.mean(train_loss))
        val_m = float(np.mean(val_loss))
        val_alloc_m = float(np.mean(val_alloc))
        val_fdsat_m = float(np.mean(val_fdsat))
        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}: train_obj={train_m:.4f}, val_obj={val_m:.4f}, "
            f"val_alloc={val_alloc_m:.4f}, val_fdsat={val_fdsat_m:.4f}, lr={cur_lr:.2e}"
        )

        if (best_val - val_m) > min_delta:
            best_val = val_m
            patience_cnt = 0
            save_path = os.path.join(model_dir, f"best_epoch{epoch}_alloc{val_alloc_m:.4f}_fdsat{val_fdsat_m:.4f}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"保存最优模型: {save_path}")
        else:
            patience_cnt += 1
            print(f"EarlyStopping计数: {patience_cnt}/{early_stop_patience}")
            if patience_cnt >= early_stop_patience:
                print("触发EarlyStopping，训练提前结束。")
                break

        # 每个 epoch 结束后推进一次学习率（无重启）
        lr_scheduler.step()

        # 参考 MAC_train_large：epoch 末显式回收
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    train()
