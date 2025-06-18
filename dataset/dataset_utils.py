from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor, Compose

import cv2
from glob import glob

class LPSD_Dataset(Dataset):
    def __init__(self, sldir, partition, imgsz=32):
        self.files = sorted(list(glob(f"{sldir}/{partition}/*/*.jpg")))
        self.imgsz = imgsz
        self.transform = Compose([
            ToTensor(),
            Resize(imgsz)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        im = cv2.imread(self.files[idx])
        im = self.transform(im)
        lb = int(self.files[idx].split("/")[-2])

        return im, lb


