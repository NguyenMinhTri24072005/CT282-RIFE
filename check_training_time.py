import os
import sys
import json
import argparse
from datetime import timedelta

# Đảm bảo in tiếng Việt có dấu mượt mà trên Windows console (cmd, powershell)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def format_seconds(seconds):
    """Định dạng số giây thành định dạng đọc được: Xh Ym Zs hoặc Xm Ys."""
    if seconds is None or seconds < 0:
        return "N/A"
    seconds = float(seconds)
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    parts = []
    if d > 0:
        parts.append(f"{d}d")
    if h > 0 or d > 0:
        parts.append(f"{h}h")
    if m > 0 or h > 0 or d > 0:
        parts.append(f"{m:02d}m" if (h > 0 or d > 0) else f"{m}m")
    parts.append(f"{s:02d}s" if parts else f"{seconds:.1f}s")
    return " ".join(parts)

def parse_model_folder(folder_name):
    """Phân tích loại mô hình, activation và attention từ tên thư mục."""
    is_modify = folder_name.startswith("modify_") or "_eca" in folder_name
    m_type = "Modify" if is_modify else "Baseline"
    
    attn = "ECA" if "eca" in folder_name else "None"
    
    # Trích xuất tên activation
    temp = folder_name
    for prefix in ["modify_eca_", "modify_", "baseline_"]:
        if temp.startswith(prefix):
            temp = temp[len(prefix):]
            break
    act = temp
    return m_type, act, attn

def load_training_info(models_dir):
    """Quét và trích xuất thông tin thời gian từ tất cả các thư mục mô hình."""
    if not os.path.exists(models_dir):
        print(f"[!] Thư mục '{models_dir}' không tồn tại.")
        return []

    records = []
    for item in sorted(os.listdir(models_dir)):
        item_path = os.path.join(models_dir, item)
        if not os.path.isdir(item_path):
            continue

        json_path = os.path.join(item_path, "experiment_results.json")
        if not os.path.exists(json_path):
            # Thử tìm file JSON có tên tương tự
            candidates = [f for f in os.listdir(item_path) if f.startswith("experiment_results") and f.endswith(".json")]
            if candidates:
                json_path = os.path.join(item_path, candidates[0])
            else:
                continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[!] Không thể đọc {json_path}: {e}")
            continue

        if not isinstance(data, list) or len(data) == 0:
            continue

        m_type, act, attn = parse_model_folder(item)
        
        epochs_run = len(data)
        times = [entry.get("epoch_time_seconds") for entry in data if isinstance(entry, dict) and entry.get("epoch_time_seconds") is not None]
        
        has_time = len(times) > 0
        total_time = sum(times) if has_time else 0.0
        avg_time = (total_time / len(times)) if has_time else 0.0
        min_time = min(times) if has_time else 0.0
        max_time = max(times) if has_time else 0.0

        # Lấy PSNR tốt nhất và epoch tương ứng
        val_psnrs = [entry.get("val_psnr", 0.0) for entry in data if isinstance(entry, dict)]
        test_psnrs = [entry.get("test_psnr", 0.0) for entry in data if isinstance(entry, dict)]
        
        best_val = max(val_psnrs) if val_psnrs else 0.0
        best_test = max(test_psnrs) if test_psnrs else 0.0
        peak_epoch = (val_psnrs.index(best_val) + 1) if val_psnrs else 0

        records.append({
            "folder": item,
            "type": m_type,
            "act": act,
            "attn": attn,
            "epochs": epochs_run,
            "has_time": has_time,
            "total_time": total_time,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "best_val_psnr": best_val,
            "best_test_psnr": best_test,
            "peak_epoch": peak_epoch
        })

    return records

def print_report(records, filter_type="all", sort_by="name"):
    """In bảng báo cáo thống kê thời gian ra console."""
    # Lọc theo loại
    if filter_type == "baseline":
        filtered = [r for r in records if r["type"] == "Baseline"]
    elif filter_type == "modify":
        filtered = [r for r in records if r["type"] == "Modify"]
    else:
        filtered = list(records)

    if not filtered:
        print("Không tìm thấy kết quả phù hợp với bộ lọc.")
        return

    # Sắp xếp
    if sort_by == "time":
        filtered.sort(key=lambda x: x["total_time"], reverse=True)
    elif sort_by == "psnr":
        filtered.sort(key=lambda x: x["best_val_psnr"], reverse=True)
    elif sort_by == "epoch":
        filtered.sort(key=lambda x: x["epochs"], reverse=True)
    elif sort_by == "type":
        filtered.sort(key=lambda x: (x["type"], x["folder"]))
    else:
        filtered.sort(key=lambda x: x["folder"])

    # Tiêu đề bảng
    header = (
        f"{'STT':<4} | {'Tên Mô Hình':<34} | {'Loại':<8} | {'Act':<24} | "
        f"{'Epochs':<6} | {'Tổng Thời Gian':<16} | {'TB/Epoch':<10} | {'Best Val PSNR':<13} | {'Đỉnh Epoch'}"
    )
    sep = "=" * len(header)
    sub_sep = "-" * len(header)

    print("\n" + sep)
    print(" BẢNG TỔNG HỢP THỜI GIAN HUẤN LUYỆN DỰ ÁN RIFE (BASELINE & MODIFY)")
    print(sep)
    print(header)
    print(sub_sep)

    total_all_time = 0.0
    total_epochs = 0
    
    for idx, r in enumerate(filtered, 1):
        total_all_time += r["total_time"]
        total_epochs += r["epochs"]
        
        t_str = format_seconds(r["total_time"])
        avg_str = f"{r['avg_time']:.1f}s" if r["has_time"] else "N/A"
        psnr_str = f"{r['best_val_psnr']:.2f} dB" if r["best_val_psnr"] > 0 else "N/A"
        peak_str = f"Ep {r['peak_epoch']}" if r["peak_epoch"] > 0 else "-"

        print(
            f"{idx:<4} | {r['folder']:<34} | {r['type']:<8} | {r['act']:<24} | "
            f"{r['epochs']:<6} | {t_str:<16} | {avg_str:<10} | {psnr_str:<13} | {peak_str}"
        )

    print(sub_sep)
    print(f"Tổng số lượt chạy hiển thị : {len(filtered)} mô hình | Tổng số Epochs: {total_epochs}")
    print(f"TỔNG THỜI GIAN HUẤN LUYỆN  : {format_seconds(total_all_time)} ({total_all_time:.1f} giây ~ {total_all_time / 3600:.2f} giờ)")
    if total_epochs > 0:
        print(f"TỐC ĐỘ TRUNG BÌNH TOÀN DỰ ÁN: {total_all_time / total_epochs:.1f} giây / Epoch")
    print(sep + "\n")

def print_pairwise_comparison(records):
    """So sánh thời gian chạy và hiệu năng giữa từng cặp Baseline và Modify (+ECA)."""
    # Gom nhóm theo activation
    baselines = {r["act"]: r for r in records if r["type"] == "Baseline"}
    modifies = {r["act"]: r for r in records if r["type"] == "Modify"}
    
    common_acts = sorted(set(baselines.keys()) | set(modifies.keys()))
    if not common_acts:
        return

    header = (
        f"{'Hàm Kích Hoạt':<24} | {'Thời Gian Baseline':<20} | {'Thời Gian Modify (+ECA)':<24} | "
        f"{'Chênh Lệch Time (Δt)':<22} | {'Δ Val PSNR'}"
    )
    sep = "=" * len(header)
    sub_sep = "-" * len(header)

    print(sep)
    print(" SO SÁNH THỜI GIAN & HIỆU NĂNG: BASELINE vs MODIFY (+ KHỐI ECA ATTENTION)")
    print(sep)
    print(header)
    print(sub_sep)

    base_times = []
    mod_times = []

    for act in common_acts:
        b = baselines.get(act)
        m = modifies.get(act)

        b_time_str = f"{format_seconds(b['total_time'])} ({b['total_time']:.0f}s)" if b else "Chưa chạy"
        m_time_str = f"{format_seconds(m['total_time'])} ({m['total_time']:.0f}s)" if m else "Chưa chạy"

        if b and m:
            base_times.append(b["total_time"])
            mod_times.append(m["total_time"])
            
            diff_time = m["total_time"] - b["total_time"]
            pct = (diff_time / b["total_time"]) * 100.0 if b["total_time"] > 0 else 0.0
            diff_str = f"{diff_time:+.0f}s ({pct:+.1f}%)"
            
            diff_psnr = m["best_val_psnr"] - b["best_val_psnr"]
            psnr_str = f"{diff_psnr:+.2f} dB"
        else:
            diff_str = "N/A"
            psnr_str = "N/A"

        print(f"{act.upper():<24} | {b_time_str:<20} | {m_time_str:<24} | {diff_str:<22} | {psnr_str}")

    print(sub_sep)
    if base_times and mod_times:
        sum_b = sum(base_times)
        sum_m = sum(mod_times)
        diff_all = sum_m - sum_b
        pct_all = (diff_all / sum_b) * 100.0 if sum_b > 0 else 0.0
        print(f"Tổng time các cặp so sánh: Baseline = {format_seconds(sum_b)} | Modify = {format_seconds(sum_m)}")
        print(f"-> Khối ECA Attention làm thay đổi thời gian huấn luyện: {diff_all:+.1f}s ({pct_all:+.2f}%)")
    print(sep + "\n")

def export_csv(records, output_path):
    """Xuất kết quả ra file CSV."""
    import csv
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Folder", "Type", "Activation", "Attention", "Epochs",
            "Total_Time_Seconds", "Total_Time_Formatted", "Avg_Time_Per_Epoch",
            "Min_Epoch_Time", "Max_Epoch_Time", "Best_Val_PSNR", "Best_Test_PSNR", "Peak_Epoch"
        ])
        for r in records:
            writer.writerow([
                r["folder"], r["type"], r["act"], r["attn"], r["epochs"],
                round(r["total_time"], 2), format_seconds(r["total_time"]), round(r["avg_time"], 2),
                round(r["min_time"], 2), round(r["max_time"], 2),
                round(r["best_val_psnr"], 4), round(r["best_test_psnr"], 4), r["peak_epoch"]
            ])
    print(f"[OK] Đã xuất báo cáo CSV thành công vào: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Kiểm tra tổng thời gian huấn luyện các mô hình Baseline và Modify trong RIFE-Project.")
    parser.add_argument("--models_dir", default="trained_model", help="Đường dẫn tới thư mục chứa các mô hình đã huấn luyện (mặc định: trained_model)")
    parser.add_argument("--filter", choices=["all", "baseline", "modify"], default="all", help="Lọc mô hình: all (tất cả), baseline, modify")
    parser.add_argument("--sort_by", choices=["name", "time", "psnr", "epoch", "type"], default="type", help="Tiêu chí sắp xếp bảng: type, name, time, psnr, epoch")
    parser.add_argument("--csv", default="", help="Đường dẫn xuất file CSV báo cáo (tùy chọn)")
    args = parser.parse_args()

    # Tự động định vị đường dẫn thư mục dự án
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    models_path = os.path.join(current_dir, args.models_dir) if not os.path.isabs(args.models_dir) else args.models_dir

    records = load_training_info(models_path)
    if not records:
        print(f"[!] Không tìm thấy dữ liệu huấn luyện nào trong: {models_path}")
        return

    print_report(records, filter_type=args.filter, sort_by=args.sort_by)
    
    if args.filter == "all":
        print_pairwise_comparison(records)

    if args.csv:
        export_csv(records, args.csv)

if __name__ == "__main__":
    main()
