from torchvision.models import vit_b_16
from ultralytics import YOLO
from torchvision.models import resnet50

from torch import nn

def create_vit(cfg):
    vit = vit_b_16(weights=cfg['vit_weights'])

    if cfg['freeze']:
        for c in vit.children():
            c.requires_grad = False

    n_ft = vit.heads.head.in_features
    class_head = nn.Sequential(nn.Linear(n_ft, 4))
    vit.heads = class_head

    return vit

def create_resnet(cfg):
    resnet = resnet50(weights=cfg['resnet_weights'])

    if cfg['freeze']:
        for c in resnet.children():
            c.requires_grad = False

    n_ft = resnet.fc.in_features
    resnet.fc = nn.Linear(n_ft, 4)

    return resnet

def create_yolo(cfg):
    yolo_name = cfg['yolo_path']
    yolo = YOLO(yolo_name)

    return yolo

