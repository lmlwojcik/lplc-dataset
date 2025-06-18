import argparse
import json

from models.yolo_model import create_yolo, train_yolo
from models.resnet_model import create_resnet, train_resnet
from models.vit_model import create_vit

import torch

def main(model_name, model_config, device):
    with open(model_config, "r") as fd:
        cfg = json.load(fd)

    if model_name == 'yolo':
        model = create_yolo(cfg)
        train_yolo(model, cfg['train_config'], cfg['data'])
    elif model_name == 'resnet':
        model = create_resnet(cfg)
        trained_model = train_resnet(model, cfg['train_config'], cfg['data'], log_cfg=cfg['log_config'])
        torch.save(trained_model.state_dict(), "v1_resnet.pth")
    elif model_name == 'vit':
        model = create_vit(cfg)
        train_resnet(model, cfg['train_config'], cfg['data'])
    else:
        print("Error -- model must be one of: yolo, resnet, vit")
        exit()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="yolo")
    parser.add_argument('--model_config', default="configs/config_yolo.json")
    #parser.add_argument('--data', default="sldir/0_1/")
    parser.add_argument('--device', default=-1)

    args = vars(parser.parse_args())
    
    main(**args)
