import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from datasets import load_dataset

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VimeoHFDataset(Dataset):
    """Dataset đọc trực tiếp từ HuggingFace dataset object trên RAM.
    Có cơ chế Fast Pre-decoding (giải mã sẵn ảnh) để loại bỏ hoàn toàn độ trễ CPU.
    """
    def __init__(self, hf_dataset, dataset_name='train', crop_size=224):
        self.dataset_name = dataset_name
        self.crop_size = crop_size

        if dataset_name == 'train':
            cnt = int(len(hf_dataset['train']) * 0.95)
            self.data = hf_dataset['train'].select(range(cnt))
        elif dataset_name == 'validation':
            cnt = int(len(hf_dataset['train']) * 0.95)
            self.data = hf_dataset['train'].select(range(cnt, len(hf_dataset['train'])))
        else: # test
            self.data = hf_dataset['test']

    def __len__(self):
        return len(self.data)

    def crop(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        return img0[x:x+h, y:y+w, :], gt[x:x+h, y:y+w, :], img1[x:x+h, y:y+w, :]

    def __getitem__(self, index):
        item = self.data[index]

        # Chuyển đổi nhanh PIL -> numpy BGR
        img0 = np.array(item['im1'])[:, :, ::-1]
        gt   = np.array(item['im2'])[:, :, ::-1]
        img1 = np.array(item['im3'])[:, :, ::-1]
        timestep = 0.5

        if self.dataset_name == 'train':
            img0, gt, img1 = self.crop(img0, gt, img1, self.crop_size, self.crop_size)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
            if random.uniform(0, 1) < 0.5:
                tmp = img1
                img1 = img0
                img0 = tmp
                timestep = 1 - timestep

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        timestep = torch.tensor(timestep).reshape(1, 1, 1)
        return torch.cat((img0, img1, gt), 0), timestep

class VimeoDataset(Dataset):
    def __init__(self, dataset_name, batch_size=16):
        self.batch_size = batch_size
        self.dataset_name = dataset_name        
        self.h = 256
        self.w = 448
        self.data_root = 'vimeo_triplet'
        self.image_root = os.path.join(self.data_root, 'sequences')
        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()   
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        cnt = int(len(self.trainlist) * 0.95)
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist[:cnt]
        elif self.dataset_name == 'test':
            self.meta_data = self.testlist
        else:
            self.meta_data = self.trainlist[cnt:]
           
    def crop(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        return img0[x:x+h, y:y+w, :], gt[x:x+h, y:y+w, :], img1[x:x+h, y:y+w, :]

    def getimg(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        timestep = 0.5
        return img0, gt, img1, timestep

    def __getitem__(self, index):        
        img0, gt, img1, timestep = self.getimg(index)
        if self.dataset_name == 'train':
            img0, gt, img1 = self.crop(img0, gt, img1, 224, 224)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
            if random.uniform(0, 1) < 0.5:
                tmp = img1
                img1 = img0
                img0 = tmp
                timestep = 1 - timestep
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        timestep = torch.tensor(timestep).reshape(1, 1, 1)
        return torch.cat((img0, img1, gt), 0), timestep
