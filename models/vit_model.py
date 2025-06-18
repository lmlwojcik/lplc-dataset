from torchvision.models import vit_b_16
from torch import nn

from dataset.dataset_utils import LPSD_Dataset

def create_vit(cfg):
    vit = vit_b_16(weights=cfg['vit_weights'])

    if cfg['freeze']:
        for c in vit.children():
            c.requires_grad = False
    print(vit)

    n_ft = vit.heads.head.in_features
    class_head = nn.Sequential(nn.Linear(n_ft, 4))
    vit.heads = class_head
    #vit = nn.Sequential(vit, class_head)
    print(vit)

    return vit

