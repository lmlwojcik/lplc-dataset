import argparse
import json
from pathlib import Path

from models.models import create_yolo, create_resnet, create_vit
from models.trainer import train_torch_model, train_yolo, test_torch_model, test_yolo

def main(config, do_train, do_test, load_model):
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
            model, results = train_yolo(model, cfg['train_config'], cfg['data'])
        else:
            model, results = train_torch_model(model, cfg['train_config'], cfg['data'])
        with open(Path(cfg['train_config']['save_path']) / "train_results.json", "w") as fd:
            json.dump(results, fd, indent=2)
    else:
        model = None
        results = {}

    if do_test:
        if model_name == 'yolo':
            test_results = test_yolo(model, cfg['test_config'], cfg['data'], "test", load_model)
        else:
            test_results = test_torch_model(model, cfg['test_config'], cfg['data'], "test", load_model)
        results.update(test_results)
        with open(Path(cfg['test_config']['save_path']) / "all_results.json", "w") as fd:
            json.dump(results, fd, indent=2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/config_yolo.json")
    parser.add_argument('--do_train', default=False, action='store_true')
    parser.add_argument('--do_test', default=False, action='store_true')
    parser.add_argument('--load_model', default=None)

    args = vars(parser.parse_args())
    
    main(**args)
