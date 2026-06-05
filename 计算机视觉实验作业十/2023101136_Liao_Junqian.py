import os
import torch
import math
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

# ========================== 1. 正弦位置编码 ==========================
def sinusoidal_pe(max_len: int, d_model: int):
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len).unsqueeze(1)
    base = 1000.0
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(base) / d_model))
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe

def plot_sinusoidal_pe(max_len=50, d_model=64):
    pe = sinusoidal_pe(max_len, d_model)
    plt.figure(figsize=(10, 6))
    plt.imshow(pe.numpy(), cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title("Sinusoidal Position Encoding (Heatmap)")
    plt.xlabel("Dimension")
    plt.ylabel("Position")
    os.makedirs("homework10/images", exist_ok=True)
    plt.savefig("homework10/images/sinusoidal_pe.png", dpi=300, bbox_inches='tight')
    plt.close()

# ========================== 2. 二维向量旋转演示 ==========================
def rotate_2d(x, y, theta):
    c, s = math.cos(theta), math.sin(theta)
    return x*c - y*s, x*s + y*c

# ========================== 3. 高维 RoPE ==========================
def rope(x: torch.Tensor, pos: torch.Tensor, d_model: int):
    assert d_model % 2 == 0
    dim_idx = torch.arange(0, d_model, 2, dtype=torch.float32)
    base = 1000.0
    theta = 1.0 / (base ** (dim_idx / d_model))
    theta = pos.unsqueeze(1) * theta.unsqueeze(0)
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    c, s = torch.cos(theta), torch.sin(theta)
    rx0 = x0 * c - x1 * s
    rx1 = x0 * s + x1 * c
    return torch.stack([rx0, rx1], dim=-1).flatten(-2)

def plot_high_dim_rope(d_model=64, seq_len=50):
    x = torch.ones(1, seq_len, d_model)*1.0
    pos = torch.arange(seq_len)
    out = rope(x, pos, d_model)
    plt.figure(figsize=(12,6))
    plt.imshow(out[0].detach().numpy(), cmap="coolwarm", vmin=-1.2, vmax=1.2)
    plt.colorbar()
    plt.title("High-Dimensional RoPE Feature Map")
    plt.xlabel("Dimension")
    plt.ylabel("Position")
    plt.savefig("homework10/images/rope_high_dim.png", dpi=300, bbox_inches='tight')
    plt.close()

# ========================== 4. E+pos和RoPE对比 ==========================
def plot_input_comparison(d_model=64, seq_len=50):
    embedding = torch.ones(1, seq_len, d_model) * 1.0
    pos = torch.arange(seq_len)
    pe = sinusoidal_pe(seq_len, d_model).unsqueeze(0)
    
    e_add = (embedding + pe)[0].numpy()
    rope_emb = rope(embedding, pos, d_model)[0].numpy()

    plt.figure(figsize=(20, 8))

    # E+pos
    plt.subplot(1, 2, 1)
    plt.imshow(e_add, cmap="coolwarm", vmin=-1.2, vmax=1.2)
    plt.title("E + Pos (Addition)", fontsize=16, fontweight='bold')
    plt.xlabel("Dimension", fontsize=12)
    plt.ylabel("Position", fontsize=12)
    plt.colorbar()

    # RoPE
    plt.subplot(1, 2, 2)
    plt.imshow(rope_emb, cmap="coolwarm", vmin=-1.2, vmax=1.2)
    plt.title("RoPE (Rotation)", fontsize=16, fontweight='bold')
    plt.xlabel("Dimension", fontsize=12)
    plt.ylabel("Position", fontsize=12)
    plt.colorbar()

    plt.suptitle("Input Method Comparison (MAX Visual Effect)", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig("homework10/images/input_comparison.png", dpi=300)
    plt.close()

# ========================== 5. RoPE相对位置 ==========================
def test_relative_position(d_model=64):
    q = torch.ones(1,1,d_model)*1.0
    k = torch.ones(1,1,d_model)*1.0
    dots = []
    labels = []

    q1 = rope(q, torch.tensor([1]), d_model)
    k1 = rope(k, torch.tensor([3]), d_model)
    dots.append((q1 @ k1.transpose(-1,-2)).item())
    labels.append("(1,3) Δ=2")

    q2 = rope(q, torch.tensor([5]), d_model)
    k2 = rope(k, torch.tensor([7]), d_model)
    dots.append((q2 @ k2.transpose(-1,-2)).item())
    labels.append("(5,7) Δ=2")

    q3 = rope(q, torch.tensor([0]), d_model)
    k3 = rope(k, torch.tensor([5]), d_model)
    dots.append((q3 @ k1.transpose(-1,-2)).item())
    labels.append("(0,5) Δ=5")

    plt.figure(figsize=(8,5))
    plt.bar(labels, dots, color=['blue','blue','orange'])
    plt.title("RoPE Relative Position Property Verification")
    plt.ylabel("Q-K Dot Product")
    plt.grid(alpha=0.3, axis='y')
    plt.savefig("homework10/images/relative_position.png", dpi=300)
    plt.close()
    return dots

# ========================== 主函数 ==========================
if __name__ == "__main__":
    plot_sinusoidal_pe()
    plot_high_dim_rope()
    plot_input_comparison()
    test_relative_position()
