from torchvision.models import resnet50
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score

from datetime import datetime
import logging

from dataset.dataset_utils import LPSD_Dataset

def create_resnet(cfg):
    resnet = resnet50(weights=cfg['resnet_weights'])

    if cfg['freeze']:
        for c in resnet.children():
            c.requires_grad = False

    n_ft = resnet.fc.in_features
    resnet.fc = nn.Linear(n_ft, 4)

    return resnet

def calc_metrics(model, data, pt="train"):
    pds = torch.tensor([])
    gts = torch.tensor([])

    with torch.no_grad():
        for i, sample in enumerate(data):
            im, lb = sample
            logits = model(im)
            lb = lb.squeeze()
            pd = logits.max(1).indices

            pds = torch.cat([pd.cpu(),pds])
            gts = torch.cat([lb.cpu(),gts])

            if pds.shape[0] > 50:
                break

    pds = pds.to(torch.int64)
    gts = gts.to(torch.int64)

    micro_f1 = multiclass_f1_score(pds,gts,average='micro')
    macro_f1 = multiclass_f1_score(pds,gts,average='macro',num_classes=4)
    acc = multiclass_accuracy(pds,gts)

    return {f"{pt}_acc": acc, f"{pt}_micro_f1": micro_f1, f"{pt}_macro_f1": macro_f1}

def dict_to_string(dct):
    ret = ""
    for k, v in dct.items():
        ret += f"| {k}: {v}"
    return ret

def train_resnet(resnet, cfg, dataset, log_cfg=None):
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
        # log = {
        #     "experiment_name": log_cfg['name'],
        #     "datetime": str(datetime.now()),
        #     "dataset size": len(train_data)
        # }
        # if "loss_log" in log_cfg.keys():
        #     log['loss_log'] = []
        # if "metric_log" in log_cfg.keys():
        #     log['loss_log'] = []

    if cfg['use_gpu'] != -1:
        resnet.to(torch.device(f"cuda:{cfg['use_gpu']}"))

    opt = Adam(resnet.named_parameters(), **cfg['optim_config'])
    loss = nn.CrossEntropyLoss()

    def train_epoch(epoch=0, step_update=-1):
        e_loss = 0
        
        for i, sample in enumerate(train_data):
            im, lb = sample
            opt.zero_grad()

            logits = resnet(im)
            c_loss = loss(logits, lb.squeeze())

            c_loss.backward()
            opt.step()
            e_loss += c_loss.item()

            if i % step_update == 0:
                print(f"Epoch {epoch} at step {i}: Loss - {c_loss.item()}")

        return e_loss/len(train_data)

    if cfg['es_metric'].endswith("loss"):
        best_metric = 1e5
    else:
        best_metric = 0
    cnt = 0
    epoch_metrics = {}

    for epoch in range(cfg['epochs']):
        print(f"Starting epoch {epoch}")

        epoch_loss = train_epoch(0, 100)
        logging.info(f"Epoch loss: {epoch_loss}")

        tm = calc_metrics(resnet, train_data, "train")
        epoch_metrics.update(tm)
        epoch_metrics['train_loss'] = epoch_loss

        if cfg['validate']: 
            vm = calc_metrics(resnet, valid_data, "val")
            epoch_metrics.update(vm)

        log_msg = "Epoch {epoch}: " dict_to_string(epoch_metrics)
        logging.info(log_msg)
        print(log_msg)

        if cfg['do_es']:
            current_metric = epoch_metrics[cfg['es_metric']]
            if (cfg['es_metric'].endswith("loss") and current_metric > best_metric) \
                    or (current_metric < best_metric):
                best_metric = current_metric
                cnt += 1
            if cnt >= cfg['patience']:
                break

    return resnet
