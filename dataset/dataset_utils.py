from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor, Compose
import torch

import cv2
from glob import glob

class LPSD_Dataset(Dataset):
    def __init__(self, sldir, partition, imgsz=32, device='cpu'):
        self.files = sorted(list(glob(f"{sldir}/{partition}/*/*.jpg")))
        self.imgsz = imgsz
        self.transform = Compose([
            ToTensor(),
            Resize(imgsz)
        ])
        if device == -1:
            device = "cpu"
        self.device = torch.device(device)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        im = cv2.imread(self.files[idx])
        im = self.transform(im)
        im = im.to(self.device)
        lb = int(self.files[idx].split("/")[-2])

        return im, lb


