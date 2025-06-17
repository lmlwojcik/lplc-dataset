from torchvision.models import resnet50
import torch.nn as nn

def create_resnet(cfg):
    resnet = resnet50(weights=cfg['resnet_weights'])

    if cfg['freeze']:
        for c in resnet.children():
            print(c)
            c.requires_grad = False

    n_ft = resnet.fc.in_features
    resnet.fc = nn.Linear(n_ft, 3)

    return resnet

def train_resnet(resnet, cfg, dataset):

    pass
