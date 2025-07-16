from torchvision.models import vit_b_16
from ultralytics import YOLO
from torchvision.models import resnet50

from torch import nn

# Workaround for the case where the torch ssl certificate expires
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class CNN_Baseline(nn.Module):
    def __init__(self, n_classes):
        super(CNN_Baseline, self).__init__()
        
        self.layer0 = nn.Conv2d(kernel_size=3, in_channels=3, out_channels=16, stride=1,padding='same')
        self.layer1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1bn = nn.BatchNorm2d(16)
        self.layer1rl = nn.ReLU()
        self.layer2 = nn.Conv2d(kernel_size=3, in_channels=16, out_channels=32, stride=1,padding='same')
        self.layer3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3bn = nn.BatchNorm2d(32)
        self.layer3rl = nn.ReLU()
        self.layer4 = nn.Conv2d(kernel_size=3, in_channels=32, out_channels=64, stride=1,padding='same')
        self.layer5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5bn = nn.BatchNorm2d(64)
        self.layer5rl = nn.ReLU()
        self.layer6 = nn.Flatten()        
        
        self.layer7 = nn.Linear(in_features=12288, out_features=128)
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
        o = self.layer4(o)
        o = self.layer5(o)
        o = self.layer5bn(o)
        o = self.layer5rl(o)
        o = self.layer6(o)
        o = self.layer7(o)
        o = self.layer7_2(o)
        o = self.layer8(o)
        return o

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

def create_base_small(cfg, n_classes=4):
    cnn = CNN_Baseline_Small(n_classes)
    return cnn

def create_baseline(cfg, n_classes=4):
    cnn = CNN_Baseline(n_classes)
    return cnn

def create_vit(cfg, n_classes=4):
    vit = vit_b_16(weights=cfg['vit_weights'])

    if cfg['freeze']:
        for c in vit.children():
            c.requires_grad = False

    n_ft = vit.heads.head.in_features
    class_head = nn.Sequential(nn.Linear(n_ft, n_classes))
    vit.heads = class_head

    return vit

def create_resnet(cfg, n_classes=4):
    resnet = resnet50(weights=cfg['resnet_weights'])

    if cfg['freeze']:
        for c in resnet.children():
            c.requires_grad = False

    n_ft = resnet.fc.in_features
    resnet.fc = nn.Linear(n_ft, n_classes)

    return resnet

def create_yolo(cfg):
    yolo_name = cfg['yolo_path']
    yolo = YOLO(yolo_name)

    return yolo

