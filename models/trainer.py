import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from ultralytics import YOLO

from datetime import datetime
from pathlib import Path
from glob import glob
import logging

from dataset.dataset_utils import LPSD_Dataset
from models.eval import calc_metrics, gen_metrics
from models.utils import dict_to_string, find_model

def train_torch_model(model, cfg, dataset, log_cfg=None):
    train_data = DataLoader(
        LPSD_Dataset(dataset['path'], "train", imgsz=dataset['imgsz'], device=cfg['use_gpu']),
        batch_size=cfg['batch_size'],
        shuffle=cfg['shuffle']
    )

    if cfg['validate']:
        valid_data = DataLoader(
            LPSD_Dataset(dataset['path'], "valid", imgsz=dataset['imgsz'], device=cfg['use_gpu']),
            batch_size=cfg['batch_size'],
            shuffle=cfg['shuffle']
        )

    if log_cfg is not None:
        logging.basicConfig(filename=f"logs/{log_cfg['experiment_name']}_{str(datetime.now())}.log")

    if cfg['use_gpu'] != -1:
        model.to(torch.device(f"cuda:{cfg['use_gpu']}"))

    opt = Adam(model.named_parameters(), **cfg['optim_config'])
    loss = nn.CrossEntropyLoss()

    def train_epoch(epoch=0, step_update=-1, c_step=0):
        e_loss = 0
        
        for i, sample in enumerate(train_data):
            c_step += 1

            im, lb = sample
            opt.zero_grad()

            logits = model(im)
            c_loss = loss(logits, lb.squeeze())

            c_loss.backward()
            opt.step()
            e_loss += c_loss.item()

            if step_update != -1 and c_step % step_update == 0:
                print(f"Epoch {epoch} at step {i}: Loss - {c_loss.item()}")

        return e_loss/len(train_data)

    if cfg['es_metric'].endswith("loss"):
        best_metric = 1e5
    else:
        best_metric = 0
    cnt = 0
    epoch_metrics = {}
    epoch = 0

    for epoch in range(1, cfg['epochs']+1):
        print(f"Starting epoch {epoch}")

        epoch_loss = train_epoch(epoch, -1)
        logging.info(f"Epoch loss: {epoch_loss}")

        tm = calc_metrics(model, train_data, "train")
        epoch_metrics.update(tm)
        epoch_metrics['train_loss'] = epoch_loss

        if cfg['validate']: 
            vm = calc_metrics(model, valid_data, "val")
            epoch_metrics.update(vm)

        log_msg = f"Epoch {epoch}: " + dict_to_string(epoch_metrics)
        logging.info(log_msg)
        print(log_msg)

        if cfg['do_es']:
            current_metric = epoch_metrics[cfg['es_metric']]
            if (cfg['es_metric'].endswith("loss") and current_metric > best_metric) \
                    or (current_metric < best_metric):
                best_metric = current_metric
                cnt += 1
                if cfg['save_best']:
                    torch.save_state_dict(model, Path(cfg['save_path']) / Path("model_best.pth"))
            if cnt >= cfg['patience']:
                break
    if cfg['save_last']:
        torch.save(model, Path(cfg['save_path']) / Path(f"model_last_epoch_{epoch}.pth"))
    if epoch == 0:
        epoch_metrics = calc_metrics(model, train_data, "train")
        vm = calc_metrics(model, valid_data, "val")
        epoch_metrics.update(vm)

    return model, epoch_metrics

def train_yolo(yolo, cfg, dataset):

    yolo.train(data=dataset['dir'], **cfg)

    return yolo, None


def test_torch_model(model, cfg, dataset, partition='test', load_model=None):
    test_data = DataLoader(
        LPSD_Dataset(dataset['path'], "test", imgsz=dataset['imgsz'], device=cfg['use_gpu']),
        batch_size=cfg['batch_size'],
        shuffle=False
    )

    if model is None:
        # We jump here without training, model must be loaded from memory
        if load_model is not None:
            model = torch.load(load_model)
        else:
            print(find_model(cfg['save_path']))
            model = torch.load(find_model(cfg['save_path']))
        if cfg['use_gpu'] != -1:
            model.to(torch.device(f"cuda:{cfg['use_gpu']}"))

    metrics = calc_metrics(model, test_data, pt=partition, return_matrix=True)
    return metrics

def test_yolo(model, cfg, dataset, partition='test', load_model=None):
    if model is None:
        if load_model is not None:
            model = YOLO(load_model)
        else:
            model = YOLO(f"{cfg['save_path']}/weights/best.pt")

    dts = LPSD_Dataset(dataset['dir'], partition, imgsz=224, device=-1)
    gts = []
    pds = []
    for g in dts.files[:50]:
        r = model(([g]))[0]

        pd = r.probs.top1
        gt = int(g.split("/")[-2])

        gts.append(gt)
        pds.append(pd)

    return gen_metrics(torch.tensor(gts,dtype=torch.int64),
                       torch.tensor(pds,dtype=torch.int64)
                       ,pt=partition,return_matrix=True)

