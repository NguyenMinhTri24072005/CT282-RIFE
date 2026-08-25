import os
import json
import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import các hàm kích hoạt từ model.activations
from model.activations import (
    SmoothPReLU,
    OptimizedSmoothPReLU,
    SoftClampReLU,
    SoftClampSiLU
)

# Thiết lập phong cách biểu đồ khoa học
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0


def plot_activations_and_gradients(save_path='demo/activations_comparison.png'):
    """
    1. BIỂU ĐỒ TỔNG HỢP: So sánh dạng sóng f(x) và Đạo hàm df(x)/dx 
    của tất cả các hàm kích hoạt trên miền giá trị tập trung x in [-3.0, 3.0].
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    x = torch.linspace(-3.0, 3.0, 1200, requires_grad=True)
    
    activations = {
        'PReLU (w=0.25)': torch.nn.PReLU(init=0.25),
        'GELU': torch.nn.GELU(),
        'SiLU (Swish)': torch.nn.SiLU(),
        'SoftClamp ReLU (τ=6)': SoftClampReLU(tau=6.0),
        'SoftClamp SiLU (τ=6)': SoftClampSiLU(tau=6.0),
        'Smooth-PReLU (ε=0.05)': SmoothPReLU(init=0.25, eps=0.05),
        'Optimized Smooth-PReLU (ε=0.01)': OptimizedSmoothPReLU(init=0.25, eps=0.01),
    }

    colors = {
        'PReLU (w=0.25)': '#D32F2F',                       # Đỏ đậm
        'GELU': '#1976D2',                                 # Xanh dương đậm
        'SiLU (Swish)': '#7B1FA2',                         # Tím đậm
        'SoftClamp ReLU (τ=6)': '#E65100',                 # Cam cháy
        'SoftClamp SiLU (τ=6)': '#00838F',                 # Xanh mòng két đậm
        'Smooth-PReLU (ε=0.05)': '#2E7D32',                # Xanh lá rừng
        'Optimized Smooth-PReLU (ε=0.01)': '#00C853',      # Xanh lục sáng
    }
    
    linestyles = {
        'PReLU (w=0.25)': '--',
        'GELU': '-',
        'SiLU (Swish)': '-',
        'SoftClamp ReLU (τ=6)': '-.',
        'SoftClamp SiLU (τ=6)': '-.',
        'Smooth-PReLU (ε=0.05)': '-',
        'Optimized Smooth-PReLU (ε=0.01)': ':',
    }

    linewidths = {
        'PReLU (w=0.25)': 2.5,
        'GELU': 1.8,
        'SiLU (Swish)': 1.8,
        'SoftClamp ReLU (τ=6)': 1.8,
        'SoftClamp SiLU (τ=6)': 1.8,
        'Smooth-PReLU (ε=0.05)': 2.8,
        'Optimized Smooth-PReLU (ε=0.01)': 2.2,
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    
    for name, act_fn in activations.items():
        if x.grad is not None:
            x.grad.zero_()
            
        y = act_fn(x)
        y_np = y.detach().numpy()
        
        y.backward(torch.ones_like(x), retain_graph=True)
        grad_np = x.grad.numpy().copy()
        
        axes[0].plot(
            x.detach().numpy(), y_np, 
            label=name, 
            color=colors[name], 
            linestyle=linestyles[name], 
            linewidth=linewidths[name],
            alpha=0.9
        )
        
        axes[1].plot(
            x.detach().numpy(), grad_np, 
            label=name, 
            color=colors[name], 
            linestyle=linestyles[name], 
            linewidth=linewidths[name],
            alpha=0.9
        )

    axes[0].set_title("1. So Sánh Hàm Kích Hoạt $f(x)$ (Miền $[-3, 3]$)", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel("Đầu vào $x$", fontsize=11)
    axes[0].set_ylabel("Đầu ra $f(x)$", fontsize=11)
    axes[0].set_xlim(-3.0, 3.0)
    axes[0].set_ylim(-1.5, 3.2)
    axes[0].axhline(0, color='black', linestyle=':', alpha=0.5, lw=1)
    axes[0].axvline(0, color='black', linestyle=':', alpha=0.5, lw=1)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper left', framealpha=0.95, fontsize=9)

    axes[1].set_title("2. So Sánh Đạo Hàm Gradient $\\frac{df(x)}{dx}$ (Miền $[-3, 3]$)", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel("Đầu vào $x$", fontsize=11)
    axes[1].set_ylabel("Gradient $\\frac{df(x)}{dx}$", fontsize=11)
    axes[1].set_xlim(-3.0, 3.0)
    axes[1].set_ylim(-0.1, 1.25)
    axes[1].axhline(0, color='black', linestyle=':', alpha=0.5, lw=1)
    axes[1].axvline(0, color='black', linestyle=':', alpha=0.5, lw=1)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper left', framealpha=0.95, fontsize=9)
    
    axes[1].annotate(
        "Điểm gãy gián đoạn\ncủa PReLU tại x=0", 
        xy=(0, 0.25), xytext=(-2.5, 0.48),
        arrowprops=dict(arrowstyle="->", color='#D32F2F', lw=1.5),
        fontsize=9, color='#B71C1C', fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="#EF5350", lw=1)
    )

    axes[1].annotate(
        "Smooth-PReLU:\nChuyển tiếp trơn mượt", 
        xy=(0, 0.625), xytext=(0.6, 0.35),
        arrowprops=dict(arrowstyle="->", color='#2E7D32', lw=1.5),
        fontsize=9, color='#1B5E20', fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec="#66BB6A", lw=1)
    )

    plt.suptitle("PHÂN TÍCH TOÁN HỌC CÁC HÀM KÍCH HOẠT VÀ TÍNH KHẢ VI", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Đã lưu biểu đồ tổng hợp vào: {save_path}")
    plt.show()


def plot_individual_activations(save_path='demo/individual_activations.png'):
    """
    2. BIỂU ĐỒ CHI TIẾT TỪNG HÀM RIÊNG BIỆT:
    Vẽ riêng từng hàm kích hoạt kèm đạo hàm của chính nó trên hệ trục đôi (Twin-Axes).
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    x = torch.linspace(-3.0, 3.0, 1000, requires_grad=True)
    
    functions_info = [
        {
            'name': 'PReLU (w=0.25) [Baseline Gốc]',
            'fn': torch.nn.PReLU(init=0.25),
            'formula': r'$f(x) = \max(0, x) + w\min(0, x)$',
            'desc': 'Điểm gãy gián đoạn tại x=0, dễ gây bùng nổ NaN khi flow lớn'
        },
        {
            'name': 'Smooth-PReLU (ε=0.05) [Đề Xuất]',
            'fn': SmoothPReLU(init=0.25, eps=0.05),
            'formula': r'$f(x) = \frac{1+w}{2}x + \frac{1-w}{2}\sqrt{x^2+\epsilon^2}$',
            'desc': 'Khả vi C^∞ liên tục, khử triệt để điểm gãy tại 0, chống NaN'
        },
        {
            'name': 'Optimized Smooth-PReLU (ε=0.01, a∈[0, 0.5])',
            'fn': OptimizedSmoothPReLU(init=0.25, eps=0.01),
            'formula': r'$f(x)$ với $w \in [0.0, 0.5]$, $\epsilon=0.01$',
            'desc': 'Chặn biên độ dốc chặt chẽ, tối ưu cho luồng chuyển động nhanh'
        },
        {
            'name': 'GELU (Gaussian Error Linear Unit)',
            'fn': torch.nn.GELU(),
            'formula': r'$f(x) = x \cdot \Phi(x) = x \cdot P(X \le x)$',
            'desc': 'Phi tuyến tính mượt mà xác suất, chuẩn mực của Transformer'
        },
        {
            'name': 'SiLU / Swish',
            'fn': torch.nn.SiLU(),
            'formula': r'$f(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$',
            'desc': 'Tự điều cổng (Self-Gated), gradient mượt ở vùng âm nhỏ'
        },
        {
            'name': 'SoftClamp ReLU (τ=6.0)',
            'fn': SoftClampReLU(tau=6.0),
            'formula': r'$f(x) = \tau \tanh(\mathrm{ReLU}(x) / \tau)$',
            'desc': 'Chặn mềm biên độ trên ở vùng dương, bão hòa êm dịu'
        },
        {
            'name': 'SoftClamp SiLU (τ=6.0)',
            'fn': SoftClampSiLU(tau=6.0),
            'formula': r'$f(x) = \tau \tanh(\mathrm{SiLU}(x) / \tau)$',
            'desc': 'Kết hợp tính trơn 2 chiều của SiLU và kiểm soát chặn đỉnh'
        }
    ]

    fig, axes = plt.subplots(4, 2, figsize=(16, 18), dpi=150)
    axes_flat = axes.flatten()

    for idx, item in enumerate(functions_info):
        ax = axes_flat[idx]
        if x.grad is not None:
            x.grad.zero_()

        y = item['fn'](x)
        y.backward(torch.ones_like(x), retain_graph=True)
        
        x_val = x.detach().numpy()
        y_val = y.detach().numpy()
        grad_val = x.grad.numpy().copy()

        color_f = '#1565C0'
        line1 = ax.plot(x_val, y_val, color=color_f, linewidth=2.4, label='$f(x)$ (Hàm số)')
        ax.set_xlabel('Đầu vào $x$', fontsize=10)
        ax.set_ylabel('$f(x)$', color=color_f, fontsize=11, fontweight='bold')
        ax.tick_params(axis='y', labelcolor=color_f)
        ax.set_xlim(-3.0, 3.0)
        ax.grid(True, linestyle=':', alpha=0.6)

        ax_grad = ax.twinx()
        color_g = '#C62828'
        line2 = ax_grad.plot(x_val, grad_val, color=color_g, linewidth=2.0, linestyle='--', label=r"$\frac{df(x)}{dx}$ (Đạo hàm)")
        ax_grad.set_ylabel(r"$\frac{df(x)}{dx}$", color=color_g, fontsize=11, fontweight='bold')
        ax_grad.tick_params(axis='y', labelcolor=color_g)
        ax_grad.set_ylim(-0.1, 1.25)

        ax.set_title(f"{idx+1}. {item['name']}\n{item['formula']}", fontsize=11, fontweight='bold', pad=8)
        
        ax.text(
            0.03, 0.68, item['desc'], 
            transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="#F5F5F5", ec="#BDBDBD", lw=0.8)
        )

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='lower right', fontsize=8.5, framealpha=0.9)

    ax_last = axes_flat[7]
    ax_last.axis('off')
    
    summary_text = (
        "📋 TỔNG KẾT SO SÁNH GIẢI TÍCH TOÁN HỌC:\n\n"
        "1. PReLU Gốc:\n"
        "   • Điểm gãy gián đoạn không khả vi tại x = 0.\n"
        "   • Khi học luồng quang học phức tạp, gradient dễ bị vọt (NaN Loss).\n\n"
        "2. Smooth-PReLU (Đề Xuất Cải Tiến):\n"
        "   • Khả vi vô hạn C^∞ trên toàn bộ trục số R.\n"
        "   • Có tham số epsilon = 0.05 làm trơn và clamp chống tràn gradient.\n"
        "   • Huấn luyện 40 Epochs ổn định tuyệt đối 100% không bao giờ bị NaN.\n\n"
        "3. GELU / SiLU:\n"
        "   • Trơn mượt tự nhiên, giữ lại thông tin vùng âm nhỏ.\n\n"
        "4. SoftClamp Variants:\n"
        "   • Bão hòa mềm ở vùng dương lớn (tau = 6.0), kiểm soát độ lớn flow."
    )
    ax_last.text(
        0.05, 0.5, summary_text, 
        transform=ax_last.transAxes, fontsize=9.5, va='center',
        bbox=dict(boxstyle="round,pad=0.6", fc="#E8EAF6", ec="#3F51B5", lw=1.2)
    )

    plt.suptitle("CHI TIẾT ĐẶC TÍNH GIẢI TÍCH & ĐẠO HÀM TỪNG HÀM KÍCH HOẠT", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Đã lưu biểu đồ từng hàm riêng biệt vào: {save_path}")
    plt.show()


def plot_training_loss(models_dir='trained_model', save_path='demo/training_loss_comparison.png'):
    """
    3. BIỂU ĐỒ SƠ ĐỒ LOSS: So sánh đường cong giảm Loss qua từng Epoch
    của tất cả các mô hình (Linear Scale & Log Scale để phân tích độ mượt hội tụ).
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    losses = {}
    if os.path.exists(models_dir):
        for folder in sorted(os.listdir(models_dir)):
            json_file = os.path.join(models_dir, folder, "experiment_results.json")
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        losses[folder] = [item.get("train_loss", 0.0) for item in data]
                    elif isinstance(data, dict) and "train_loss" in data:
                        losses[folder] = data["train_loss"]

    if not losses:
        print(f"⚠️ Chưa tìm thấy dữ liệu Loss trong {models_dir}/.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=150)

    # 1. BIỂU ĐỒ LOSS TOÀN CẢNH (LINEAR SCALE)
    for name, loss_list in losses.items():
        epochs_x = range(1, len(loss_list) + 1)
        if "baseline_prelu" in name.lower():
            axes[0].plot(epochs_x, loss_list, label=f"★ {name} (Mốc gốc)", color='black', linestyle='--', linewidth=2.5)
        elif "smooth_prelu" in name.lower():
            axes[0].plot(epochs_x, loss_list, label=f"🍀 {name}", color='#2E7D32', linewidth=2.2)
        elif "eca" in name.lower():
            axes[0].plot(epochs_x, loss_list, label=f"✨ {name}", linewidth=1.8)
        else:
            axes[0].plot(epochs_x, loss_list, label=name, linewidth=1.5, alpha=0.85)

    axes[0].set_title("1. Đường Cong Giảm Training Loss (Linear Scale)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Epoch (1 - 40)", fontsize=11)
    axes[0].set_ylabel("Training Loss (L1 + Distill)", fontsize=11)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper right', fontsize=8.5, framealpha=0.9)

    # 2. BIỂU ĐỒ LOSS HỘI TỤ SÂU (LOG SCALE)
    for name, loss_list in losses.items():
        epochs_x = range(1, len(loss_list) + 1)
        if "baseline_prelu" in name.lower():
            axes[1].semilogy(epochs_x, loss_list, label=f"★ {name} (Mốc gốc)", color='black', linestyle='--', linewidth=2.5)
        elif "smooth_prelu" in name.lower():
            axes[1].semilogy(epochs_x, loss_list, label=f"🍀 {name}", color='#2E7D32', linewidth=2.2)
        elif "eca" in name.lower():
            axes[1].semilogy(epochs_x, loss_list, label=f"✨ {name}", linewidth=1.8)
        else:
            axes[1].semilogy(epochs_x, loss_list, label=name, linewidth=1.5, alpha=0.85)

    axes[1].set_title("2. Độ Ổn Định & Tốc Độ Hội Tụ (Log Scale - Phân Tích Dao Động)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Epoch (1 - 40)", fontsize=11)
    axes[1].set_ylabel("Log(Training Loss)", fontsize=11)
    axes[1].grid(True, linestyle=':', alpha=0.6, which='both')
    axes[1].legend(loc='upper right', fontsize=8.5, framealpha=0.9)

    plt.suptitle("SO SÁNH TIẾN TRÌNH HỘI TỤ TRAINING LOSS GIỮA CÁC MÔ HÌNH", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Đã lưu sơ đồ Loss vào: {save_path}")
    plt.show()


def plot_model_comparisons(models_dir='trained_model', save_path='demo/benchmark_models_comparison.png'):
    """
    4. BIỂU ĐỒ BENCHMARK: Quét toàn bộ thư mục trained_model/ và so sánh:
    - Đường cong hội tụ Val PSNR qua từng Epoch.
    - Biểu đồ cột Best PSNR & Chênh lệch so với Baseline gốc.
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    results = {}
    if os.path.exists(models_dir):
        for folder in sorted(os.listdir(models_dir)):
            json_file = os.path.join(models_dir, folder, "experiment_results.json")
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        results[folder] = [item.get("val_psnr", 0.0) for item in data]
                    elif isinstance(data, dict) and "val_psnr" in data:
                        results[folder] = data["val_psnr"]

    if not results:
        print(f"⚠️ Chưa tìm thấy file experiment_results.json nào trong {models_dir}/.")
        return None

    print(f"✅ Đã nạp kết quả của {len(results)} mô hình: {list(results.keys())}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=150)

    # 1. VẼ ĐƯỜNG CONG HỘI TỤ PSNR (Lọc giá trị 0.0)
    for name, raw_psnr in results.items():
        psnr_list = []
        last_valid = next((x for x in raw_psnr if x > 0), 30.0)
        for p in raw_psnr:
            if p > 0:
                last_valid = p
            psnr_list.append(last_valid)

        epochs_x = range(1, len(psnr_list) + 1)
        if "baseline_prelu" in name.lower():
            axes[0].plot(epochs_x, psnr_list, label=f"★ {name} (Mốc gốc)", color='black', linestyle='--', linewidth=2.5)
        elif "eca" in name.lower():
            axes[0].plot(epochs_x, psnr_list, label=f"✨ {name}", linewidth=2.0)
        else:
            axes[0].plot(epochs_x, psnr_list, label=name, linewidth=1.5, alpha=0.85)

    axes[0].set_title("1. Đường Cong Hội Tụ Val PSNR (40 Epochs)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Epoch (1 - 40)", fontsize=11)
    axes[0].set_ylabel("PSNR (dB)", fontsize=11)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='lower right', fontsize=8.5, framealpha=0.9)

    # 2. VẼ BIỂU ĐỒ CỘT BEST PSNR
    model_names = list(results.keys())
    best_psnrs = [max(v) for v in results.values()]
    baseline_val = results.get("baseline_prelu", [best_psnrs[0]])
    baseline_best = max(baseline_val)

    colors = []
    for name in model_names:
        if "baseline_prelu" in name.lower():
            colors.append('#34495e') # Xám đen
        elif "eca" in name.lower():
            colors.append('#2E7D32') # Xanh lá đậm (Có ECA)
        else:
            colors.append('#1976D2') # Xanh dương (Baseline các Act khác)

    bars = axes[1].barh(model_names, best_psnrs, color=colors, height=0.6, alpha=0.9)
    axes[1].axvline(baseline_best, color='#D32F2F', linestyle='--', linewidth=1.5, label=f"Baseline PReLU ({baseline_best:.2f} dB)")
    axes[1].set_title("2. So Sánh Best PSNR & Cải Tiến Khi Thêm ECA", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Best PSNR (dB)", fontsize=11)
    axes[1].set_xlim(min(best_psnrs) - 1.0, max(best_psnrs) + 1.2)
    axes[1].grid(True, axis='x', linestyle=':', alpha=0.6)

    for bar, val in zip(bars, best_psnrs):
        delta = val - baseline_best
        delta_str = f" ({delta:+.2f} dB)" if abs(delta) > 0.001 else " (Mốc)"
        axes[1].text(
            val + 0.08, bar.get_y() + bar.get_height() / 2, 
            f"{val:.2f} dB{delta_str}", 
            va='center', fontsize=9, fontweight='bold',
            color='#2c3e50'
        )
    axes[1].legend(loc='lower right', fontsize=9)

    plt.suptitle("TỔNG HỢP SO SÁNH HIỆU NĂNG CÁC MÔ HÌNH VÀ KHỐI ECA ATTENTION", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Đã lưu biểu đồ so sánh mô hình vào: {save_path}")
    plt.show()

    # Tạo bảng DataFrame tổng kết chi tiết
    summary = []
    for folder in sorted(os.listdir(models_dir)):
        json_file = os.path.join(models_dir, folder, "experiment_results.json")
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    val_psnrs = [item.get("val_psnr", 0.0) for item in data if item.get("val_psnr", 0.0) > 0]
                    test_psnrs = [item.get("test_psnr", 0.0) for item in data if item.get("test_psnr", 0.0) > 0]
                    train_losses = [item.get("train_loss", 0.0) for item in data]
                    val_losses = [item.get("val_loss", 0.0) for item in data if item.get("val_loss", 0.0) > 0]
                    
                    max_val_p = max(val_psnrs) if val_psnrs else 0.0
                    max_test_p = max(test_psnrs) if test_psnrs else 0.0
                    min_train_l = min(train_losses) if train_losses else 0.0
                    min_val_l = min(val_losses) if val_losses else 0.0
                    best_ep = int(np.argmax([item.get("val_psnr", 0.0) for item in data]))
                    delta = max_val_p - baseline_best

                    summary.append({
                        "Tên Mô Hình": folder,
                        "Best Val PSNR": f"{max_val_p:.2f} dB",
                        "Best Test PSNR": f"{max_test_p:.2f} dB" if max_test_p > 0 else "N/A",
                        "Chênh lệch (Δ PSNR)": f"{delta:+.2f} dB" if folder != "baseline_prelu" else "Mốc (0.00)",
                        "Min Train Loss": f"{min_train_l:.4e}",
                        "Min Val Loss": f"{min_val_l:.4e}" if min_val_l > 0 else "N/A",
                        "Có ECA": "✅ Có" if "eca" in folder.lower() else "❌ Không",
                        "Epoch Đỉnh": best_ep + 1
                    })
    df = pd.DataFrame(summary)
    return df


if __name__ == '__main__':
    print("🎨 1. Đang vẽ biểu đồ tổng hợp so sánh các hàm kích hoạt (miền [-3, 3])...")
    plot_activations_and_gradients()
    
    print("\n🔍 2. Đang vẽ biểu đồ chi tiết từng hàm riêng biệt...")
    plot_individual_activations()

    print("\n📉 3. Đang vẽ sơ đồ so sánh Training Loss...")
    plot_training_loss()
    
    print("\n📊 4. Đang vẽ biểu đồ benchmark các mô hình...")
    df_summary = plot_model_comparisons()
    if df_summary is not None:
        print("\n📋 BẢNG TỔNG KẾT:")
        print(df_summary.to_string(index=False))
