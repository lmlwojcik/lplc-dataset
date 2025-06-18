from torchvision.models import resnet50
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.dataset_utils import LPSD_Dataset

def create_resnet(cfg):
    resnet = resnet50(weights=cfg['resnet_weights'])

    if cfg['freeze']:
        for c in resnet.children():
            c.requires_grad = False

    n_ft = resnet.fc.in_features
    resnet.fc = nn.Linear(n_ft, 4)

    return resnet

def train_resnet(resnet, cfg, dataset):
    opt = Adam(resnet.named_parameters(), **cfg['optim_config'])
    loss = nn.CrossEntropyLoss()

    train_data = DataLoader(LPSD_Dataset(dataset['path'], "train", imgsz=dataset['imgsz']),
                      batch_size=cfg['batch_size'], shuffle=cfg['shuffle'])

    def train_epoch():
        e_loss = 0
        
        for i, sample in enumerate(train_data):
            im, lb = sample
            opt.zero_grad()

            logits = resnet(im)
            c_loss = loss(logits, lb)

            c_loss.backward()
            opt.step()
            e_loss += c_loss.item()
            if i % 20 == 0:
                print(c_loss.item())

        return e_loss

    print(train_epoch())

    exit()
    pass
