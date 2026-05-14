import numpy as np
import matplotlib.pyplot as plt

def check_gradients(model, plot_grad_dist=True):
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"⚠️ [梯度丢失] {name}: 未接收梯度")
        else:
            grad_norm = param.grad.norm().item()
            print(f"✅ [梯度正常] {name}: 梯度范数={grad_norm:.4e}")

    # 可视化梯度分布
    if plot_grad_dist:
        all_grads = []
        for param in model.parameters():
            if param.grad is not None:
                all_grads.append(param.grad.detach().view(-1).cpu().numpy())

        if len(all_grads) > 0:
            all_grads = np.concatenate(all_grads)
            plt.figure(figsize=(10, 6))
            plt.hist(all_grads, bins=100, alpha=0.7)
            plt.yscale('log')
            plt.xlabel("Gradient Value")
            plt.ylabel("Frequency (log scale)")
            plt.title("Gradient Distribution")
            plt.grid(True, which="both", ls="--")
            # **保存为图片**
            plt.savefig("gradient_distribution.png")  # 避免 plt.show() 的问题
            print("📊 梯度分布图已保存为 gradient_distribution.png")
        else:
            print("⚠️ 所有参数均无梯度，无法绘制分布图")
