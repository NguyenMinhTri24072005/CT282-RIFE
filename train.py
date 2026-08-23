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

def get_learning_rate(step, args):
    if step < 2000:
        mul = step / 2000.
        return 3e-4 * mul
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return (3e-4 - 3e-6) * mul + 3e-6

def evaluate(model, val_data, device):
    psnr_list = []
    for i, data in enumerate(val_data):
        data_gpu, timestep = data
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.        
        imgs = data_gpu[:, :6]
        gt = data_gpu[:, 6:9]
        with torch.no_grad():
            pred, info = model.update(imgs, gt, training=False)
        for j in range(gt.shape[0]):
            psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
            psnr_list.append(psnr)
    return np.array(psnr_list).mean()

def train(model, args, device):
    step = 0
    start_epoch = 0
    best_psnr = 0.0
    results_log = []

    if args.resume and os.path.exists(args.resume):
        print(f"[*] Phục hồi từ Checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint['best_psnr']
        if hasattr(model.flownet, 'module'):
            model.flownet.module.load_state_dict(checkpoint['model_state'])
        else:
            model.flownet.load_state_dict(checkpoint['model_state'])
        model.optimG.load_state_dict(checkpoint['optimizer_state'])
        
        json_path = os.path.join(args.save_dir, 'experiment_results.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                results_log = json.load(f)

    # NẠP DATASET TRỰC TIẾP TỪ HUGGINGFACE HOẶC LOCAL
    if os.path.exists('vimeo_triplet/sequences'):
        print("[*] Nạp dataset từ thư mục vimeo_triplet cục bộ...")
        dataset = VimeoDataset('train')
        dataset_val = VimeoDataset('validation')
    else:
        print(f"[*] Nạp dataset trực tiếp từ Hugging Face ({args.hf_dataset}) vào RAM...")
        hf_ds = load_dataset(args.hf_dataset)
        dataset = VimeoHFDataset(hf_ds, 'train')
        dataset_val = VimeoHFDataset(hf_ds, 'validation')

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

    if args.local_rank in [-1, 0]:
        print(f"🚀 Bắt đầu huấn luyện {args.epoch} Epochs | Số step/epoch: {args.step_per_epoch} | Multi-GPU: {args.is_distributed} | Batch: {args.batch_size}...")

    for epoch in range(start_epoch, args.epoch):
        t_epoch_start = time.time()
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        loss_epoch = []
        for i, data in enumerate(train_data):
            data_gpu, timestep = data
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
            
            learning_rate = get_learning_rate(step, args)
            for param_group in model.optimG.param_groups:
                param_group['lr'] = learning_rate
                
            pred, info = model.update(imgs, gt, learning_rate, training=True)
            loss_epoch.append(info['loss_l1'].item())
            
            if args.local_rank in [-1, 0] and (step % 20 == 0 or i == args.step_per_epoch - 1):
                print('Epoch: {} {}/{} | LR: {:.6f} | Loss: {:.4e}'.format(epoch, i, args.step_per_epoch, learning_rate, info['loss_l1']))
            step += 1

        epoch_duration = time.time() - t_epoch_start

        if args.local_rank in [-1, 0]:
            # ĐÁNH GIÁ ĐỊNH KỲ THEO EVAL_INTERVAL HOẶC EPOCH CUỐI CÙNG
            should_eval = ((epoch + 1) % args.eval_interval == 0) or (epoch == args.epoch - 1)
            val_psnr = best_psnr
            
            if should_eval:
                t_eval_start = time.time()
                val_psnr = evaluate(model, val_data, device)
                eval_duration = time.time() - t_eval_start
                print(f"=== Đánh giá Epoch {epoch}: PSNR = {val_psnr:.2f} dB (Mất {eval_duration:.1f}s) | Thời gian train: {epoch_duration:.1f}s ===")
            else:
                print(f"=== Hoàn thành Epoch {epoch} trong {epoch_duration:.1f}s ===")
            
            results_log.append({
                "epoch": epoch,
                "train_loss": np.mean(loss_epoch),
                "val_psnr": float(val_psnr)
            })
            with open(os.path.join(args.save_dir, 'experiment_results.json'), 'w') as f:
                json.dump(results_log, f, indent=4)

            resume_path = os.path.join(args.save_dir, "latest_resume_state.pth")
            state_to_save = {
                'epoch': epoch,
                'model_state': model.flownet.module.state_dict() if hasattr(model.flownet, 'module') else model.flownet.state_dict(),
                'optimizer_state': model.optimG.state_dict(),
                'best_psnr': best_psnr
            }
            torch.save(state_to_save, resume_path)
            
            if val_psnr > best_psnr and should_eval:
                best_psnr = val_psnr
                best_model_path = os.path.join(args.save_dir, "best_flownet.pkl")
                torch.save(model.flownet.module.state_dict() if hasattr(model.flownet, 'module') else model.flownet.state_dict(), best_model_path)
                print(f"🌟 Đã phá kỷ lục! Lưu mô hình tốt nhất vào {best_model_path}")
                
        if args.is_distributed:
            dist.barrier()

    if args.local_rank in [-1, 0]:
        resume_path = os.path.join(args.save_dir, "latest_resume_state.pth")
        if os.path.exists(resume_path):
            os.remove(resume_path)
            print("🧹 Hoàn tất huấn luyện. Đã xóa phao cứu sinh (latest_resume_state.pth).")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=40, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--eval_interval', default=2, type=int, help='Khoảng cách epoch để chạy validation')
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
