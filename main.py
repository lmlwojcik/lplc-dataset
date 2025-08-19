import argparse
from glob import glob
import json
from pathlib import Path

from models.models import (
    create_yolo,
    create_resnet,
    create_vit,
    create_baseline
)

from models.trainer import (
    train_torch_model,
    train_yolo,
    test_torch_model,
    test_yolo,
    predict_torch_model,
    predict_yolo
)

def main(cfg, model_cfg, train_cfg, test_cfg, # Overall configs
         do_predict, partition, load_model,   # For gathering predictions
         dataset, run_cluster, run_name, n_features        # Per-run experiment variables
    ):
    
    cfg['save_path'] += "/" + dataset.replace("/", "_")
    #if run_cluster is not None:
    #    cfg['save_path'] += "/" + run_cluster
    if run_name is not None:
        print(f"Starting run: {run_name}")
        if cfg['model_name'] != 'yolo':
            cfg['save_path'] += "/" + run_name
        else:
            cfg["name"] = run_name
            if train_cfg is not None:
                train_cfg['name'] = run_name
    else:
        print("Starting run")

    tag = "path" if cfg['model_name'] != 'yolo' else "dir"
    if dataset is not None:
        cfg['data'][tag] = dataset
    print(f"Dataset partition: {cfg['data'][tag]}")
    n_classes = len(glob(f"{dataset}/train/*"))
    cfg['n_classes'] = n_classes
    print(f"This protocol has {n_classes} classes.")

    model_name = cfg['model_name']

    if model_name == 'yolo':
        model = create_yolo(cfg)
    elif model_name == 'resnet':
        model = create_resnet(cfg, n_classes)
    elif model_name == 'vit':
        model = create_vit(cfg, n_classes)
    elif model_name == 'small':
        cfg['model_cfg'] = model_cfg
        cfg['n_features'] = n_features
        cfg['n_classes'] = n_classes
        model = create_baseline(cfg['model_cfg'], n_features, n_classes)
    else:
        print("Error -- model must be one of: yolo, resnet, vit, base, small. Got: ", model_name)
        exit()

    # Train
    if train_cfg is not None:
        if model_name == 'yolo':
            model, results = train_yolo(model, train_cfg, cfg['data'], cfg['save_path'])
        else:
            model, results = train_torch_model(model, train_cfg, cfg['data'], cfg['save_path'], cfg['log_config'])
        if results is not None:
            with open(Path(cfg['save_path']) / "train_results.json", "w") as fd:
                json.dump(results, fd, indent=2)
        else:
            results = {}
    else:
        model = None
        results = {}

    # Test or Evaluate
    if test_cfg is not None:
        if model_name == 'yolo':
            test_results = test_yolo(model, test_cfg, cfg['data'], cfg, partition, load_model)
            cfg['save_path'] += "/" + cfg['name']
        else:
            test_results = test_torch_model(model, test_cfg, cfg['data'], cfg, partition, load_model, n_classes)
        results.update(test_results)
        print(json.dumps(results, indent=2))
        with open(Path(cfg['save_path']) / "all_results.json", "w") as fd:
            json.dump(results, fd, indent=2)

    # Test and get predictions
    if do_predict:
        if model_name == 'yolo':
            predict_results = predict_yolo(model, cfg['data'], cfg, partition, load_model)
            if test_cfg is None:
                cfg['save_path'] += "/" + cfg['name']
        else:
            predict_results = predict_torch_model(model, cfg['data'], cfg, partition, load_model, n_classes)
        print(predict_results['metrics'])
        with open(Path(cfg['save_path']) / f"predict_results_{partition}_with_logits.json", "w") as fd:
            json.dump(predict_results, fd, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', default="configs/config_yolo.json", type=str)
    parser.add_argument('-mc', '--model_config', default=None, type=str)

    parser.add_argument('-d', '--device', default='cpu', type=str)
    parser.add_argument('-bs', '--batch_size', default=None, type=int)

    parser.add_argument('-t', '--train_config', default=None, type=str)
    parser.add_argument('-v', '--test_config', default=None, type=str)

    parser.add_argument('-p', '--do_predict', default=False, action='store_true')
    parser.add_argument('-pt', '--partition', default="test", type=str)
    parser.add_argument('-n', '--run_name', default=None, type=str)
    parser.add_argument('-rc', '--run_cluster', default=None, type=str)

    parser.add_argument('-dt', '--dataset', default=None, type=str)
    parser.add_argument('-m', '--load_model', default=None, type=str)
    parser.add_argument('-nf', '--n_features', default=None, type=int)

    clargs = vars(parser.parse_args())

    config = clargs['config']
    with open(config, "r") as fd:
        cfg = json.load(fd)
    
    if clargs['train_config'] is None:
        if 'train_config' in cfg.keys():
            train_cfg = config['train_config']
        else:
            train_cfg = None
    else:
        with open(clargs['train_config'], "r") as fd:
            train_cfg = json.load(fd)

    if clargs['test_config'] is None:
        if 'test_config' in cfg.keys():
            test_cfg = config['test_config']
        else:
            test_cfg = None
    else:
        with open(clargs['test_config'], "r") as fd:
            test_cfg = json.load(fd)

    def update_cfgs(tr_cfg, te_cfg, clarg, cfg_name, arg_name):
        if tr_cfg is not None:
            tr_cfg[cfg_name] = clarg[arg_name]
        if te_cfg is not None:
            te_cfg[cfg_name] = clarg[arg_name]

    if clargs['device'] is not None:
        if len(clargs['device']) == 1:
            clargs['device'] = f"cuda:{clargs['device']}"
        update_cfgs(train_cfg, test_cfg, clargs,
                        "use_gpu" if cfg['model_name'] != "yolo" else "device", 'device')
        cfg['use_gpu'] = clargs['device']

    if clargs['batch_size'] is not None:
        update_cfgs(train_cfg, test_cfg, clargs,
                        "batch_size" if cfg['model_name'] != "yolo" else "batch", 'batch_size')

    args = {
        'cfg': cfg,
        'model_cfg': clargs['model_config'],
        'train_cfg': train_cfg,
        'test_cfg': test_cfg,

        'do_predict': clargs['do_predict'],
        'partition': clargs['partition'],
        'load_model': clargs['load_model'],
        'dataset': clargs['dataset'],
        'run_name': clargs['run_name'],
        'run_cluster': clargs['run_cluster'],
        'n_features': clargs['n_features']
    }

    if args['dataset'] is None:
        args['dataset'] = cfg['data']['path']

    main(**args)
