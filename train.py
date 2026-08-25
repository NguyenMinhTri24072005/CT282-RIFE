import os
import math
import time
import json
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset

from dataset import VimeoHFDataset, VimeoDataset

def format_time(seconds):
    """Format thời gian thành chuỗi dễ đọc (Xh Ym Zs hoặc Xm Ys)."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{seconds:.1f}s"

def get_learning_rate(step, args):
    if step < 2000:
        mul = step / 2000.
        return 3e-4 * mul
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return (3e-4 - 3e-6) * mul + 3e-6

def evaluate(model, data_loader, device):
    """
    Đánh giá độ chính xác (PSNR) và Loss trên tập dữ liệu được truyền vào (Val hoặc Test).
    Trả về: (mean_psnr, mean_loss)
    """
    psnr_list = []
    loss_list = []
    
    with torch.no_grad():
        for data in data_loader:
            data_gpu, timestep = data
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.        
            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
            
            pred, info = model.update(imgs, gt, training=False)
            loss_list.append(info['loss_l1'].item())
            
            for j in range(gt.shape[0]):
                diff = gt[j] - pred[j]
                mse = torch.mean(diff * diff)
                if mse > 0:
                    psnr = -10.0 * math.log10(mse.item())
                    psnr_list.append(psnr)
                    
    mean_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    mean_loss = float(np.mean(loss_list)) if loss_list else 0.0
    return mean_psnr, mean_loss

def train(model, args, device):
    step = 0
    start_epoch = 1 # Bắt đầu đếm Epoch từ 1
    best_psnr = 0.0
    results_log = []
    total_start_time = time.time()

    if args.resume and os.path.exists(args.resume):
        print(f"[*] Phục hồi từ Checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint.get('best_psnr', 0.0)
        if hasattr(model.flownet, 'module'):
            model.flownet.module.load_state_dict(checkpoint['model_state'])
        else:
            model.flownet.load_state_dict(checkpoint['model_state'])
        model.optimG.load_state_dict(checkpoint['optimizer_state'])
        
        json_path = os.path.join(args.save_dir, 'experiment_results.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                results_log = json.load(f)

    # NẠP DATASET TRỰC TIẾP TỪ HUGGINGFACE HOẶC LOCAL CHO CẢ TRAIN, VAL VÀ TEST
    if os.path.exists('vimeo_triplet/sequences'):
        print("[*] Nạp dataset từ thư mục vimeo_triplet cục bộ...")
        dataset = VimeoDataset('train')
        dataset_val = VimeoDataset('validation')
        dataset_test = VimeoDataset('test')
    else:
        print(f"[*] Nạp dataset trực tiếp từ Hugging Face ({args.hf_dataset}) vào RAM...")
        hf_ds = load_dataset(args.hf_dataset)
        dataset = VimeoHFDataset(hf_ds, 'train')
        dataset_val = VimeoHFDataset(hf_ds, 'validation')
        dataset_test = VimeoHFDataset(hf_ds, 'test')

    sampler = DistributedSampler(dataset) if args.is_distributed else None
    
    # TỐI ƯU HÓA DATALOADER VỚI PERSISTENT WORKERS VÀ PREFETCH
    train_data = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=True, 
        sampler=sampler, 
        shuffle=(sampler is None),
        persistent_workers=True,
        prefetch_factor=2
    )
    args.step_per_epoch = train_data.__len__()
    val_data = DataLoader(dataset_val, batch_size=32, pin_memory=True, num_workers=2)
    test_data = DataLoader(dataset_test, batch_size=32, pin_memory=True, num_workers=2)

    if args.local_rank in [-1, 0]:
        print(f"🚀 Bắt đầu huấn luyện {args.epoch} Epochs (Từ Epoch {start_epoch} đến {args.epoch}) | Steps/Epoch: {args.step_per_epoch} | Multi-GPU: {args.is_distributed} | Batch: {args.batch_size}...")

    last_val_psnr = 0.0
    last_val_loss = 0.0
    last_test_psnr = 0.0
    last_test_loss = 0.0

    # VÒNG LẶP CHẠY TỪ 1 ĐẾN ARGS.EPOCH
    for epoch in range(start_epoch, args.epoch + 1):
        t_epoch_start = time.time()
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        loss_epoch = []
        loss_tea_epoch = []
        loss_distill_epoch = []
        lr_epoch = 0.0

        for i, data in enumerate(train_data):
            data_gpu, timestep = data
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
            
            learning_rate = get_learning_rate(step, args)
            lr_epoch = learning_rate
            for param_group in model.optimG.param_groups:
                param_group['lr'] = learning_rate
                
            pred, info = model.update(imgs, gt, learning_rate, training=True)
            
            loss_epoch.append(info['loss_l1'].item())
            if 'loss_tea' in info:
                loss_tea_epoch.append(info['loss_tea'].item())
            if 'loss_distill' in info:
                loss_distill_epoch.append(info['loss_distill'].item())
            
            if args.local_rank in [-1, 0] and (step % 20 == 0 or i == args.step_per_epoch - 1):
                print(f"Epoch: {epoch}/{args.epoch} {i+1}/{args.step_per_epoch} | LR: {learning_rate:.6f} | Loss: {info['loss_l1'].item():.4e} | Loss_Tea: {info.get('loss_tea', torch.tensor(0.0)).item():.4e} | Distill: {info.get('loss_distill', torch.tensor(0.0)).item():.4e}")
            step += 1

        epoch_duration = time.time() - t_epoch_start
        total_elapsed = time.time() - total_start_time
        completed_epochs = epoch - start_epoch + 1
        avg_epoch_time = total_elapsed / completed_epochs
        remaining_epochs = args.epoch - epoch
        eta_seconds = remaining_epochs * avg_epoch_time

        avg_train_loss = float(np.mean(loss_epoch)) if loss_epoch else 0.0
        avg_loss_tea = float(np.mean(loss_tea_epoch)) if loss_tea_epoch else 0.0
        avg_loss_distill = float(np.mean(loss_distill_epoch)) if loss_distill_epoch else 0.0

        if args.local_rank in [-1, 0]:
            # ĐÁNH GIÁ ĐỊNH KỲ: Epoch 1 (đầu tiên), các Epoch chia hết cho eval_interval, hoặc Epoch cuối cùng
            should_eval = (epoch == 1) or (epoch % args.eval_interval == 0) or (epoch == args.epoch)
            
            if should_eval:
                t_eval_start = time.time()
                val_psnr, val_loss = evaluate(model, val_data, device)
                test_psnr, test_loss = evaluate(model, test_data, device)
                eval_duration = time.time() - t_eval_start
                
                last_val_psnr = val_psnr
                last_val_loss = val_loss
                last_test_psnr = test_psnr
                last_test_loss = test_loss
            else:
                val_psnr = last_val_psnr
                val_loss = last_val_loss
                test_psnr = last_test_psnr
                test_loss = last_test_loss

            is_record = (val_psnr > best_psnr) and should_eval
            if is_record:
                best_psnr = val_psnr

            # IN THÔNG TIN TỪNG EPOCH (BẮT ĐẦU TỪ 1)
            print("\n" + "=" * 88)
            print(f"📊 [EPOCH {epoch}/{args.epoch}] HOÀN TẤT TRONG {format_time(epoch_duration)} (Đã chạy: {format_time(total_elapsed)} | Còn lại: ~{format_time(eta_seconds)})")
            print(f"• Train Metrics : Loss = {avg_train_loss:.4e} | Loss_Tea = {avg_loss_tea:.4e} | Distill = {avg_loss_distill:.4e} | LR = {lr_epoch:.6f}")
            print(f"• Val Metrics   : PSNR = {val_psnr:.2f} dB | Val Loss = {val_loss:.4e}")
            print(f"• Test Metrics  : PSNR = {test_psnr:.2f} dB | Test Loss = {test_loss:.4e}")
            if is_record:
                print(f"🌟 [KỶ LỤC MỚI]: Đạt Best Val PSNR = {best_psnr:.2f} dB! Đang lưu model tốt nhất...")
            else:
                print(f"ℹ️ Best Val PSNR hiện tại: {best_psnr:.2f} dB")
            print("=" * 88 + "\n")
            
            # LƯU EPOCH 1-BASED VÀO FILE JSON
            results_log.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "loss_tea": avg_loss_tea,
                "loss_distill": avg_loss_distill,
                "learning_rate": lr_epoch,
                "val_psnr": float(val_psnr),
                "val_loss": float(val_loss),
                "test_psnr": float(test_psnr),
                "test_loss": float(test_loss),
                "is_best": bool(is_record),
                "best_psnr": float(best_psnr),
                "epoch_time_seconds": epoch_duration
            })
            
            with open(os.path.join(args.save_dir, 'experiment_results.json'), 'w') as f:
                json.dump(results_log, f, indent=4)

            # LƯU PHAO CỨU SINH RESUME STATE
            resume_path = os.path.join(args.save_dir, "latest_resume_state.pth")
            state_to_save = {
                'epoch': epoch,
                'model_state': model.flownet.module.state_dict() if hasattr(model.flownet, 'module') else model.flownet.state_dict(),
                'optimizer_state': model.optimG.state_dict(),
                'best_psnr': best_psnr
            }
            torch.save(state_to_save, resume_path)
            
            # LƯU MODEL ĐẠT KỶ LỤC MỚI
            if is_record:
                best_model_path = os.path.join(args.save_dir, "best_flownet.pkl")
                torch.save(model.flownet.module.state_dict() if hasattr(model.flownet, 'module') else model.flownet.state_dict(), best_model_path)
                print(f"💾 Đã lưu trọng số best model vào: {best_model_path}")
                
        if args.is_distributed:
            dist.barrier()

    if args.local_rank in [-1, 0]:
        total_training_time = time.time() - total_start_time
        print("\n" + "=" * 65)
        print(f"🎉 HOÀN THÀNH HUẤN LUYỆN TOÀN BỘ {args.epoch} EPOCHS (EPOCH 1 -> {args.epoch})!")
        print(f"⏱️  TỔNG THỜI GIAN HUẤN LUYỆN : {format_time(total_training_time)} ({total_training_time:.1f} giây)")
        print(f"⚡ TỐC ĐỘ TRUNG BÌNH        : {total_training_time / (args.epoch - start_epoch + 1):.1f}s / Epoch")
        print(f"🏆 BEST VAL PSNR ĐẠT ĐƯỢC   : {best_psnr:.2f} dB")
        print("=" * 65 + "\n")

        resume_path = os.path.join(args.save_dir, "latest_resume_state.pth")
        if os.path.exists(resume_path):
            os.remove(resume_path)
            print("🧹 Đã xóa phao cứu sinh (latest_resume_state.pth).")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=40, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--eval_interval', default=2, type=int, help='Khoảng cách epoch để chạy validation & test')
    parser.add_argument('--local_rank', default=-1, type=int, help='local rank cho DDP, -1 là Single GPU')
    parser.add_argument('--model_type', type=str, choices=['original', 'modify'], default='original')
    parser.add_argument('--act', type=str, default='prelu')
    parser.add_argument('--attn', type=str, default='none')
    parser.add_argument('--hf_dataset', type=str, default='bijinc/vimeo-90k-mini', help='Tên dataset trên Hugging Face')
    parser.add_argument('--save_dir', type=str, default='trained_model/baseline_prelu')
    parser.add_argument('--resume', type=str, default='')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # TỰ ĐỘNG NHẬN DIỆN SINGLE GPU HOẶC MULTI-GPU (DDP)
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1:
        args.local_rank = env_local_rank

    args.is_distributed = (args.local_rank != -1)

    if args.is_distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    if args.model_type == 'original':
        from model_original.RIFE import Model
        model = Model(args.local_rank)
    else:
        from model.RIFE import Model
        model = Model(args.local_rank, act_name=args.act, attn_name=args.attn)
        
    train(model, args, device)
