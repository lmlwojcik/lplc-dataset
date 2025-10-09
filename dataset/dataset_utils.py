from torch.utils.data import Dataset, BatchSampler
from torchvision.transforms import ToTensor, Compose
import torch

import random
import math
import os
import cv2
from glob import glob


def resize_with_pad(image, 
                    new_shape, 
                    padding_color = (127, 127, 127)):
    h, w = image.shape[:2]
    nh,nw = new_shape[:2]
    
    sw = nw / w
    sh = nh / h
    scale = min(sw, sh)

    if w > nw or h > nh:
        new_w = int(w * scale)
        new_h = int(h * scale)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        new_w = w
        new_h = h

    delta_w = nw - new_w
    delta_h = nh - new_h
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image

def calc_class_weights(gts):
    gt = torch.tensor(gts)
    n_cls = len(torch.unique(gt))
    freq = torch.bincount(gt, minlength=n_cls)

    ws = 1.0 / freq
    ws = ws / ws.sum()
    return ws

class LPSD_Dataset(Dataset):
    def __init__(self, sldir, partition, grayscale=True, imgsz=32, device='cpu'):
        if device == -1:
            device = "cpu"
        self.device = torch.device(device)
        self.files = sorted(list(glob(f"{sldir}/{partition}/*/*")))
        if isinstance(imgsz, int):
            imgsz = (imgsz, imgsz)
        self.imgsz = imgsz
        self.transform = Compose([
            ToTensor()
        ])
        self.ims = []
        self.lbt = []
        self.gts = []
        self.fs = []
        self.nfs = 0
        for f in self.files:
            lb = int(f.split("/")[-2])
            self.gts.append(lb)
            if not os.path.exists(f):
                continue
            self.nfs += 1
            im = cv2.imread(f)
            if grayscale:
                im = cv2.cvtColor(im, cv2.BGR2GRAY)
            im = resize_with_pad(cv2.imread(f), imgsz)
            self.fs.append(f)
            self.ims.append(self.transform(im).to(self.device))
            self.lbt.append(torch.tensor([lb]).to(self.device))
        self.cls_weights = calc_class_weights(self.gts)

    def __len__(self):
        return self.nfs

    def __getitem__(self, idx):
        im = self.ims[idx]
        lb = self.lbt[idx]
        return im, lb

class BalancedSampler(BatchSampler):
    def __init__(self, data, bs, cls, static=False, static_shuffle=True):
        self.data = data
        self.bs = bs
        self.n_cls = cls
        self.map = {c: [] for c in range(cls)}
        for i in range(len(data)):
            lb = data[i][1].item()
            self.map[lb].append(i)
        mn = min([len(x) for x in self.map.values()])
        self.n = mn*self.n_cls

        self.static = static
        self.static_shuffle = static_shuffle
        if self.static:
            for v in self.map.values():
                random.shuffle(v)
            mn = min([len(x) for x in self.map.values()])
            self.idxs = []
            for v in self.map.values():
                self.idxs += v[:mn]
            if self.static_shuffle:
                random.shuffle(self.idxs)

    def __iter__(self):
        if self.static:
            if not self.static_shuffle:
                random.shuffle(self.idxs)
            for i in range(self.__len__()):
                yield self.idxs[i*self.bs:(i+1)*self.bs]
            return

        mn = min([len(x) for x in self.map.values()])
        for v in self.map.values():
            random.shuffle(v)

        # Batches possible unbalanced:
        idxs = []
        for v in self.map.values():
            idxs += v[:mn]
        random.shuffle(idxs)

        for i in range(math.ceil((mn*len(self.map.keys())) / self.bs)):
            yield idxs[i*self.bs:(i+1)*self.bs]

        # All batches guaranteed to be balanced:
        # for i in range(math.ceil((mn*len(self.map.keys())) / self.bs)):
        #     rt = []
        #     for j in self.map.keys():
        #         rt += self.map[j][i*(self.bs//len(self.map.keys())):(i+1)*(self.bs//len(self.map.keys()))]
        #     random.shuffle(rt)
        #     yield rt


    def __len__(self):
        return math.ceil(self.n / self.bs)
    