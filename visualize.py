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
    Vẽ so sánh chi tiết dạng sóng đầu ra f(x) và Đạo hàm Gradient df(x)/dx 
    của tất cả các hàm kích hoạt (PReLU, GELU, SiLU, SoftClamp, SmoothPReLU).
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # Tạo dải giá trị đầu vào x từ -4.0 đến 4.0 với autograd
    x = torch.linspace(-4.0, 4.0, 1000, requires_grad=True)
    
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
        'PReLU (w=0.25)': '#e74c3c',                       # Đỏ
        'GELU': '#3498db',                                 # Xanh dương
        'SiLU (Swish)': '#9b59b6',                         # Tím
        'SoftClamp ReLU (τ=6)': '#e67e22',                 # Cam
        'SoftClamp SiLU (τ=6)': '#1abc9c',                 # Xanh ngọc
        'Smooth-PReLU (ε=0.05)': '#2ecc71',                # Xanh lá đậm
        'Optimized Smooth-PReLU (ε=0.01)': '#27ae60',      # Xanh lá sáng
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

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    
    for name, act_fn in activations.items():
        # Xóa gradient cũ
        if x.grad is not None:
            x.grad.zero_()
            
        # 1. Tính f(x)
        y = act_fn(x)
        y_np = y.detach().numpy()
        
        # 2. Tính đạo hàm df(x)/dx bằng Autograd
        y.backward(torch.ones_like(x), retain_graph=True)
        grad_np = x.grad.numpy().copy()
        
        # Vẽ hàm kích hoạt f(x)
        axes[0].plot(
            x.detach().numpy(), y_np, 
            label=name, 
            color=colors[name], 
            linestyle=linestyles[name], 
            linewidth=2.2 if 'Smooth' in name else 1.8
        )
        
        # Vẽ Gradient df(x)/dx
        axes[1].plot(
            x.detach().numpy(), grad_np, 
            label=name, 
            color=colors[name], 
            linestyle=linestyles[name], 
            linewidth=2.2 if 'Smooth' in name else 1.8
        )

    # Cấu hình đồ thị f(x)
    axes[0].set_title("1. So Sánh Hàm Kích Hoạt $f(x)$", fontsize=14, fontweight='bold', pad=12)
    axes[0].set_xlabel("Đầu vào $x$", fontsize=12)
    axes[0].set_ylabel("Đầu ra $f(x)$", fontsize=12)
    axes[0].axhline(0, color='gray', linestyle=':', alpha=0.6)
    axes[0].axvline(0, color='gray', linestyle=':', alpha=0.6)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper left', framealpha=0.9, fontsize=9.5)

    # Cấu hình đồ thị Gradient df(x)/dx
    axes[1].set_title("2. So Sánh Đạo Hàm Gradient $\\frac{df(x)}{dx}$", fontsize=14, fontweight='bold', pad=12)
    axes[1].set_xlabel("Đầu vào $x$", fontsize=12)
    axes[1].set_ylabel("Gradient $\\frac{df(x)}{dx}$", fontsize=12)
    axes[1].axhline(0, color='gray', linestyle=':', alpha=0.6)
    axes[1].axvline(0, color='gray', linestyle=':', alpha=0.6)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper left', framealpha=0.9, fontsize=9.5)
    
    # Chú thích điểm gãy của PReLU vs SmoothPReLU
    axes[1].annotate(
        "Điểm gãy gián đoạn\ncủa PReLU tại x=0", 
        xy=(0, 0.25), xytext=(-2.8, 0.45),
        arrowprops=dict(arrowstyle="->", color='#e74c3c', lw=1.5),
        fontsize=9, color='#c0392b', fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2e9", ec="#e74c3c", lw=1)
    )

    plt.suptitle("PHÂN TÍCH TOÁN HỌC CÁC HÀM KÍCH HOẠT VÀ TÍNH KHẢ VI", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Đã lưu biểu đồ hàm kích hoạt vào: {save_path}")
    plt.show()


def plot_model_comparisons(models_dir='trained_model', save_path='demo/benchmark_models_comparison.png'):
    """
    Quét toàn bộ thư mục trained_model/ và vẽ biểu đồ so sánh:
    1. Đường cong hội tụ Val PSNR qua từng Epoch.
    2. Biểu đồ cột Best PSNR & Chênh lệch so với Baseline gốc.
    3. So sánh trước và sau khi bổ sung khối ECA Attention.
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
                        results[folder] = [item["val_psnr"] for item in data]
                    elif isinstance(data, dict) and "val_psnr" in data:
                        results[folder] = data["val_psnr"]

    if not results:
        print(f"⚠️ Chưa tìm thấy file experiment_results.json nào trong {models_dir}/.")
        return None

    print(f"✅ Đã nạp kết quả của {len(results)} mô hình: {list(results.keys())}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=150)

    # 1. VẼ ĐƯỜNG CONG HỘI TỤ PSNR
    for name, psnr_list in results.items():
        if "baseline_prelu" in name.lower():
            axes[0].plot(psnr_list, label=f"★ {name} (Mốc gốc)", color='black', linestyle='--', linewidth=2.5)
        elif "eca" in name.lower():
            axes[0].plot(psnr_list, label=f"✨ {name}", linewidth=2.0)
        else:
            axes[0].plot(psnr_list, label=name, linewidth=1.5, alpha=0.85)

    axes[0].set_title("1. Đường Cong Hội Tụ Val PSNR (40 Epochs)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Epoch", fontsize=11)
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
            colors.append('#27ae60') # Xanh lá (Có ECA)
        else:
            colors.append('#3498db') # Xanh dương (Baseline các Act khác)

    bars = axes[1].barh(model_names, best_psnrs, color=colors, height=0.6, alpha=0.9)
    axes[1].axvline(baseline_best, color='#e74c3c', linestyle='--', linewidth=1.5, label=f"Baseline PReLU ({baseline_best:.2f} dB)")
    axes[1].set_title("2. So Sánh Best PSNR & Cải Tiến Khi Thêm ECA", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Best PSNR (dB)", fontsize=11)
    axes[1].set_xlim(min(best_psnrs) - 1.0, max(best_psnrs) + 1.2)
    axes[1].grid(True, axis='x', linestyle=':', alpha=0.6)

    # Ghi số điểm và delta lên từng cột
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

    # Tạo bảng DataFrame tổng kết
    summary = []
    for name, psnr_list in results.items():
        max_p = max(psnr_list)
        delta = max_p - baseline_best
        summary.append({
            "Tên Mô Hình": name,
            "Best PSNR (dB)": f"{max_p:.2f}",
            "Chênh lệch (Δ PSNR)": f"{delta:+.2f} dB" if name != "baseline_prelu" else "Mốc chuẩn (0.00)",
            "Có ECA Attention": "✅ Có" if "eca" in name.lower() else "❌ Không",
            "Epoch Đạt Đỉnh": int(np.argmax(psnr_list))
        })
    df = pd.DataFrame(summary)
    return df


if __name__ == '__main__':
    print("🎨 Đang vẽ biểu đồ toán học các hàm kích hoạt...")
    plot_activations_and_gradients()
    
    print("\n📊 Đang vẽ biểu đồ so sánh các mô hình...")
    df_summary = plot_model_comparisons()
    if df_summary is not None:
        print("\n📋 BẢNG TỔNG KẾT:")
        print(df_summary.to_string(index=False))
