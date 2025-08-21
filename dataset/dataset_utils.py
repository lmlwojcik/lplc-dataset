from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose
import torch

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
    def __init__(self, sldir, partition, imgsz=32, device='cpu'):
        if device == -1:
            device = "cpu"
        self.device = torch.device(device)
        self.files = sorted(list(glob(f"{sldir}/{partition}/*/*")))
        self.imgsz = imgsz
        self.transform = Compose([
            ToTensor()
        ])
        self.ims = []
        self.lbt = []
        self.gts = []
        self.nfs = 0
        for f in self.files:
            lb = int(f.split("/")[-2])
            self.gts.append(lb)
            if not os.path.exists(f):
                continue
            self.nfs += 1
            im = resize_with_pad(cv2.imread(f), imgsz)
            self.ims.append(self.transform(im).to(self.device))
            self.lbt.append(torch.tensor([lb]).to(self.device))
        self.cls_weights = calc_class_weights(self.gts)

    def __len__(self):
        return self.nfs

    def __getitem__(self, idx):
        im = self.ims[idx]
        lb = self.lbt[idx]
        return im, lb


