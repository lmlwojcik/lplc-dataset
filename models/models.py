from torchvision.models import vit_b_16, vit_b_32, vit_l_16
from ultralytics import YOLO
from torchvision.models import resnet50, resnet101, resnet152, resnet18, resnet34
from models.ocr_archs import make_GPLPR

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import FeaturePyramidNetwork as FPN

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
    if cfg['model_cfg'] == 'resnet18':
        resnet = resnet18(weights=cfg['resnet_weights'])
    if cfg['model_cfg'] == 'resnet34':
        resnet = resnet34(weights=cfg['resnet_weights'])

    if cfg['freeze']:
        for c in resnet.children():
            c.requires_grad = False

    n_ft = resnet.fc.in_features
    if "task" in cfg.keys() and cfg['task'] == "regression":
        n_classes = 1
        resnet.fc = nn.Linear(n_ft, n_classes)
        resnet = nn.Sequential(resnet, nn.Sigmoid())
    else:
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
        backbone = ocr_model
        #backbone = nn.Sequential(encoder, att, nn.Flatten())
        in_channels = 64*7
    else:
        backbone = ocr_model.encoder
        #backbone = nn.Sequential(encoder, nn.Flatten())
        in_channels = 64*(cfg['data']['imgsz'][0]//4)*(cfg['data']['imgsz'][1]//4)

    if cfg['freeze']:
        for c in backbone.children():
            c.requires_grad = False
    else:
        for c in backbone.children():
            c.requires_grad = True
    
    class OCRLeg(nn.Module):
        def __init__(self, n_cls):
            super().__init__()
            self.l1 = nn.Linear(192, 64)
            self.l1a = nn.ReLU()
            self.l2 = nn.Linear(192, 64)
            self.l2a = nn.ReLU()
            self.l3 = nn.Linear(192, 64)
            self.l3a = nn.ReLU()
            self.l4 = nn.Linear(192, 64)
            self.l4a = nn.ReLU()
            self.l5 = nn.Linear(192, 64)
            self.l5a = nn.ReLU()
            self.l6 = nn.Linear(192, 64)
            self.l6a = nn.ReLU()
            self.l7 = nn.Linear(192, 64)
            self.l7a = nn.ReLU()

            self.h1 = nn.Linear(64*7, 192)
            self.h1a = nn.ReLU()
            self.h2 = nn.Linear(192, 64)
            self.h2a = nn.ReLU()
            self.cls = nn.Linear(64, n_cls)

        def forward(self, input):
            x = input[0]

            x1 = self.l1(x.select(1, 0))
            x1 = self.l1a(x1)
            x2 = self.l2(x.select(1, 1))
            x2 = self.l2a(x2)
            x3 = self.l3(x.select(1, 2))
            x3 = self.l3a(x3)
            x4 = self.l4(x.select(1, 3))
            x4 = self.l4a(x4)
            x5 = self.l5(x.select(1, 4))
            x5 = self.l5a(x5)
            x6 = self.l6(x.select(1, 5))
            x6 = self.l6a(x6)
            x7 = self.l7(x.select(1, 6))
            x7 = self.l7a(x7)

            nx = torch.cat([x1,x2,x3,x4,x5,x6,x7], dim=1)
        
            x = self.h1(nx)
            x = self.h1a(x)
            x = self.h2(x)
            x = self.h2a(x)
            out = self.cls(x)

            return out

    #hidden_size = cfg['hidden_size']
    #fc_channels = cfg['fc_channels']

    # head = nn.Sequential(
    #     nn.Linear(in_channels, hidden_size),
    #     nn.ReLU(),
    #     nn.Linear(hidden_size, fc_channels),
    #     nn.ReLU(),
    #     nn.Linear(fc_channels, n_classes),
    #     #nn.Softmax()
    # )
    model = nn.Sequential(backbone, OCRLeg(4))

    return model


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        factor = 2 if bilinear else 1
        self.down3 = (Down(256, 512//factor))
        self.up1 = (Up(512, 256 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up3 = (Up(128, 64, bilinear))

        self.fpn = FPN([64, 128, 256], 16)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(196608, 256)
        self.fc2 = nn.Linear(49152, 256)
        self.fc3 = nn.Linear(12288, 256)
        self.hidden = nn.Linear(256*3, 192)
        self.out = nn.Linear(192, 4)

        # self.fco1 = nn.Linear(32*64*192, 192)
        # self.fco2 = nn.Linear(192, n_classes)
        #self.outc = (OutConv(192, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x_up1 = self.up1(x4, x3)
        #print(x_up1.shape)
        x_up2 = self.up2(x_up1, x2)
        #print(x_up2.shape)
        x_up3 = self.up3(x_up2, x1)
        #print(x_up3.shape)
        
        fpn_out = self.fpn({k:v for k,v in enumerate([x_up3, x_up2, x_up1])})

        #print(self.flatten(fpn_out[0]).shape, self.flatten(fpn_out[1]).shape, self.flatten(fpn_out[2]).shape)
        f1 = self.fc1(self.flatten(fpn_out[0]))
        f2 = self.fc2(self.flatten(fpn_out[1]))
        f3 = self.fc3(self.flatten(fpn_out[2]))
        catd = torch.cat([f1,f2,f3], dim=1)
        hd = self.hidden(catd)
        logits = self.out(hd)

        #x = self.flatten(x_up4)
        #x = self.fc1(x)
        #logits = self.fc2(x)
        #logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

def create_unet(cfg, n_classes):
    unet = UNet(3, n_classes)
    print(unet)
    return unet

def get_model_with_weights(cfg, load_model, device, n_classes=4):
    if cfg['model_name'] == 'small':
        model = create_baseline(cfg['model_cfg'], cfg['n_features'], cfg['n_classes'])
    elif cfg['model_name'] == 'resnet':
        model = create_resnet(cfg, n_classes=n_classes)
    elif cfg['model_name'] == 'yolo':
        model = create_yolo(cfg, n_classes=n_classes, torch_training=True)
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
