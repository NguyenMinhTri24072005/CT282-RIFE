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

from dataset import VimeoDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_learning_rate(step, args):
    if step < 2000:
        mul = step / 2000.
        return 3e-4 * mul
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return (3e-4 - 3e-6) * mul + 3e-6

def evaluate(model, val_data):
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

def train(model, args):
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

    dataset = VimeoDataset('train')
    sampler = DistributedSampler(dataset) if args.world_size > 1 else None
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, drop_last=True, sampler=sampler, shuffle=(sampler is None))
    args.step_per_epoch = train_data.__len__()
    dataset_val = VimeoDataset('validation')
    val_data = DataLoader(dataset_val, batch_size=16, pin_memory=True, num_workers=4)

    print('Bắt đầu huấn luyện...')
    for epoch in range(start_epoch, args.epoch):
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
            
            if args.local_rank == 0 and step % 100 == 0:
                print('Epoch: {} {}/{} | LR: {:.6f} | Loss: {:.4e}'.format(epoch, i, args.step_per_epoch, learning_rate, info['loss_l1']))
            step += 1

        if args.local_rank == 0:
            val_psnr = evaluate(model, val_data)
            print(f"=== Đánh giá Epoch {epoch}: PSNR = {val_psnr:.2f} dB ===")
            
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
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_model_path = os.path.join(args.save_dir, "best_flownet.pkl")
                torch.save(model.flownet.module.state_dict() if hasattr(model.flownet, 'module') else model.flownet.state_dict(), best_model_path)
                print(f"🌟 Đã phá kỷ lục! Lưu mô hình tốt nhất vào {best_model_path}")
                
        if args.world_size > 1:
            dist.barrier()

    if args.local_rank == 0:
        resume_path = os.path.join(args.save_dir, "latest_resume_state.pth")
        if os.path.exists(resume_path):
            os.remove(resume_path)
            print("🧹 Hoàn tất huấn luyện. Đã xóa phao cứu sinh (latest_resume_state.pth).")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=40, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--model_type', type=str, choices=['original', 'modify'], default='original')
    parser.add_argument('--act', type=str, default='prelu')
    parser.add_argument('--attn', type=str, default='none')
    parser.add_argument('--save_dir', type=str, default='trained_model/baseline_prelu')
    parser.add_argument('--resume', type=str, default='')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.world_size > 1:
        torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
        torch.cuda.set_device(args.local_rank)
        
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
        
    train(model, args)
