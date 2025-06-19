import argparse
import json

from models.models import create_yolo, create_resnet, create_vit
from models.trainer import train_torch_model, train_yolo

def main(config, do_train, do_test):
    with open(config, "r") as fd:
        cfg = json.load(fd)
    model_name = cfg['model_name']

    if model_name == 'yolo':
        model = create_yolo(cfg)
    elif model_name == 'resnet':
        model = create_resnet(cfg)
    elif model_name == 'vit':
        model = create_vit(cfg)
    else:
        print("Error -- model must be one of: yolo, resnet, vit")
        exit()

    if do_train:
        if model_name == 'yolo':
            model = train_yolo(model, cfg['train_config'], cfg['data'])
        else:
            model = train_torch_model(model, cfg['train_config'], cfg['data'])

    if do_test:
        if model_name == 'yolo':
            model = train_yolo(model, cfg['train_config'], cfg['data'])
        else:
            model = train_torch_model(model, cfg['train_config'], cfg['data'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/config_yolo.json")
    parser.add_argument('--do_train', default=False, action='store_true')
    parser.add_argument('--do_test', default=False, action='store_true')

    args = vars(parser.parse_args())
    
    main(**args)
