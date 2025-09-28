from torchvision.models import vit_b_16, vit_b_32, vit_l_16
from ultralytics import YOLO
from torchvision.models import resnet50, resnet101, resnet152
from models.ocr_archs import make_GPLPR

import torch
from torch import nn

from models.utils import find_model
from models.builder import build_network
import yaml

# Workaround for the case where the torch ssl certificate expires
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class CNN_Baseline_Small(nn.Module):
    def __init__(self, n_classes):
        super(CNN_Baseline_Small, self).__init__()
        
        self.layer0 = nn.Conv2d(kernel_size=3, in_channels=3, out_channels=16, stride=1,padding='same')
        self.layer1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1bn = nn.BatchNorm2d(16)
        self.layer1rl = nn.ReLU()
        self.layer2 = nn.Conv2d(kernel_size=3, in_channels=16, out_channels=32, stride=1,padding='same')
        self.layer3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3bn = nn.BatchNorm2d(32)
        self.layer3rl = nn.ReLU()
        self.layer6 = nn.Flatten()
        
        self.layer7 = nn.Linear(in_features=24576, out_features=128)
        self.layer7_2 = nn.ReLU()
        self.layer8 = nn.Linear(in_features=128, out_features=n_classes)

    def forward(self, x):
        o = self.layer0(x)
        o = self.layer1(o)
        o = self.layer1bn(o)
        o = self.layer1rl(o)
        o = self.layer2(o)
        o = self.layer3(o)
        o = self.layer3bn(o)
        o = self.layer3rl(o)
        o = self.layer6(o)
        o = self.layer7(o)
        o = self.layer7_2(o)
        o = self.layer8(o)
        return o

def create_baseline(model_cfg, n_features=128, n_classes=4):
    if model_cfg is not None:
        with open(model_cfg, 'r') as fd:
            ncfg = yaml.safe_load(fd)
        return build_network(ncfg['architecture'], n_fts=n_features, n_cls=n_classes)
    cnn = CNN_Baseline_Small(n_classes)
    return cnn

def create_vit(cfg, n_classes=4):
    if cfg['model_cfg'] == 'vit_b_16':
        vit = vit_b_16(weights=cfg['vit_weights'])
    elif cfg['model_cfg'] == 'vit_b_32':
        vit = vit_b_32(weights=cfg['vit_weights'])
    elif cfg['model_cfg'] == 'vit_l_16':
        vit = vit_l_16(weights=cfg['vit_weights'])

    if cfg['freeze']:
        for c in vit.children():
            c.requires_grad = False

    n_ft = vit.heads.head.in_features
    class_head = nn.Sequential(nn.Linear(n_ft, n_classes))
    vit.heads = class_head

    return vit

def create_resnet(cfg, n_classes=4):
    if cfg['model_cfg'] == 'resnet50':
        resnet = resnet50(weights=cfg['resnet_weights'])
    elif cfg['model_cfg'] == 'resnet101':
        resnet = resnet101(weights=cfg['resnet_weights'])
    elif cfg['model_cfg'] == 'resnet152':
        resnet = resnet152(weights=cfg['resnet_weights'])

    if cfg['freeze']:
        for c in resnet.children():
            c.requires_grad = False

    n_ft = resnet.fc.in_features
    resnet.fc = nn.Linear(n_ft, n_classes)

    return resnet

def create_yolo(cfg, n_classes=4, torch_training=False):
    yolo_name = cfg['yolo_path']
    yolo = YOLO(yolo_name)
    if not torch_training:
        return yolo
    else:
        bb = yolo
        backbone = nn.Sequential(*list(bb.model.model.children())[:-1])  # shared feature extractor without the Classify block
        
        out_channels = list(bb.model.model.children())[-1].conv.conv.out_channels
        in_channels = list(bb.model.model.children())[-1].conv.conv.in_channels

        head = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(out_channels, n_classes)
            )
        yolo = nn.Sequential(backbone, head)
    
    return yolo

def create_ocr_encoder(cfg, n_classes=4):
    ocr_model = make_GPLPR()
    ws = torch.load(cfg['ocr_weights'])
    ocr_model.load_state_dict(ws['model']['sd'])

    encoder = ocr_model.encoder
    att = ocr_model.attention
    if cfg['use_attention']:
        backbone = nn.Sequential(encoder, att, nn.Flatten())
        in_channels = 64*7
    else:
        backbone = nn.Sequential(encoder, nn.Flatten())
        in_channels = 64*(cfg['data']['imgsz'][0]//4)*(cfg['data']['imgsz'][1]//4)

    for c in backbone.children():
        c.requires_grad = False
    hidden_size = cfg['hidden_size']
    fc_channels = cfg['fc_channels']

    head = nn.Sequential(
        nn.Linear(in_channels, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, fc_channels),
        nn.ReLU(),
        nn.Linear(fc_channels, n_classes),
        #nn.Softmax()
    )
    model = nn.Sequential(backbone, head)

    return model


def get_model_with_weights(cfg, load_model, device, n_classes=4):
    if cfg['model_name'] == 'small':
        model = create_baseline(cfg['model_cfg'], cfg['n_features'], cfg['n_classes'])
    elif cfg['model_name'] == 'resnet':
        model = create_resnet(cfg, n_classes=n_classes)
    else:
        model = create_vit(cfg, n_classes=n_classes)
    
    if load_model is not None:
        ckpt = torch.load(load_model)
    else:
        ckpt = torch.load(find_model(cfg['save_path']))
    model.load_state_dict(ckpt)

    if device != -1:
        if len(device) == 1:
            device = f"cuda:{device}"
        model.to(torch.device(f"{device}"))
    return model
