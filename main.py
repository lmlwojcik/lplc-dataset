import argparse
from glob import glob
import json
from pathlib import Path

from models.models import create_yolo, create_resnet, create_vit, create_baseline, create_base_small
from models.trainer import train_torch_model, train_yolo, test_torch_model, test_yolo, predict_torch_model

def main(config, do_train, do_test, do_predict, partition, load_model, dataset, run_name):
    with open(config, "r") as fd:
        cfg = json.load(fd)
    if run_name is not None:
        print(f"Starting run: {run_name}")
        if cfg['model_name'] != 'yolo':
            tag = "save_path"
            cfg['train_config'][tag] += "/" + run_name
            cfg['test_config']['save_path'] += "/" + run_name
        else:
            tag = "name"
            cfg['train_config'][tag] = run_name
            cfg['test_config'][tag] = run_name
            cfg['test_config']['save_path'] += "/" + run_name
    else:
        print("Starting run")
    tag = "path" if cfg['model_name'] != 'yolo' else "dir"
    if dataset is not None:
        cfg['data'][tag] = dataset
    print(f"Dataset partition: {cfg['data'][tag]}")
    n_classes = len(glob(f"{dataset}/train/*"))
    print(f"This protocol has {n_classes} classes.")

    model_name = cfg['model_name']

    if model_name == 'yolo':
        model = create_yolo(cfg)
    elif model_name == 'resnet':
        model = create_resnet(cfg, n_classes)
    elif model_name == 'vit':
        model = create_vit(cfg, n_classes)
    elif model_name == 'base':
        model = create_baseline(cfg, n_classes)
    elif model_name == 'small':
        model = create_base_small(cfg, n_classes)
    else:
        print("Error -- model must be one of: yolo, resnet, vit, base, small")
        exit()

    if do_train:
        if model_name == 'yolo':
            model, results = train_yolo(model, cfg['train_config'], cfg['data'], cfg['save_path'])
        else:
            model, results = train_torch_model(model, cfg['train_config'], cfg['data'], cfg['log_config'])
        if results is not None:
            with open(Path(cfg['train_config']['save_path']) / "train_results.json", "w") as fd:
                json.dump(results, fd, indent=2)
        else:
            results = {}
    else:
        model = None
        results = {}

    if do_test:
        if model_name == 'yolo':
            test_results = test_yolo(model, cfg['train_config'], cfg['data'], partition, load_model)
        else:
            test_results = test_torch_model(model, cfg['test_config'], cfg['data'], partition, load_model)
        results.update(test_results)
        print(json.dumps(results, indent=2))
        with open(Path(cfg['test_config']['save_path']) / "all_results.json", "w") as fd:
            json.dump(results, fd, indent=2)

    if do_predict:
        predict_results = predict_torch_model(model, cfg['test_config'], cfg['data'], partition, load_model)
        print(predict_results['metrics'])
        with open(Path(cfg['test_config']['save_path']) / f"predict_results_{partition}.json", "w") as fd:
            json.dump(predict_results, fd, indent=2)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default="configs/config_yolo.json")
    parser.add_argument('-t', '--do_train', default=False, action='store_true')
    parser.add_argument('-v', '--do_test', default=False, action='store_true')
    parser.add_argument('-p', '--do_predict', default=False, action='store_true')
    parser.add_argument('-pt', '--partition', default="test")
    parser.add_argument('-n', '--run_name', default=None)
    parser.add_argument('-d', '--dataset', default=None)
    parser.add_argument('-m', '--load_model', default=None)

    args = vars(parser.parse_args())
    
    main(**args)
