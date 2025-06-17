import argparse
from models.yolo_model import create_yolo, train_yolo
import json

def main(model_name, model_config, data, device):
    with open(model_config, "r") as fd:
        cfg = json.load(fd)

    if model_name == 'yolo':
        model = create_yolo(cfg)
    elif model_name == 'resnet':
        model = None
    elif model_name == 'vit':
        model = None
    else:
        print("Error -- model must be one of: yolo, resnet, vit")
        exit()
    
    train_yolo(model, cfg['train_config'], data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="yolo")
    parser.add_argument('--model_config', default="configs/config_yolo.json")
    parser.add_argument('--data', default="sldir/0_1/")
    parser.add_argument('--device', default=-1)

    args = vars(parser.parse_args())
    
    main(**args)