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

def calc_metrics(model, data):
    pds = torch.tensor([])
    gts = torch.tensor([])

    with torch.no_grad():
        for i, sample in enumerate(data):
            im, lb = sample
            logits = model(im)
            lb = lb.squeeze()
            pd = logits.max(1).indices

            pds = torch.cat([pd,pds])
            gts = torch.cat([lb,gts])

            if pds.shape[0] > 50:
                break

    pds = pds.to(torch.int64)
    gts = gts.to(torch.int64)

    micro_f1 = multiclass_f1_score(pds,gts,average='micro')
    macro_f1 = multiclass_f1_score(pds,gts,average='macro',num_classes=4)
    acc = multiclass_accuracy(pds,gts)

    return {"acc": acc, "micro_f1": micro_f1, "macro_f1": macro_f1}


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

    for epoch in range(cfg['epochs']):
        print(f"Starting epoch {epoch}")

        epoch_loss = train_epoch(0, 100)
        tm = calc_metrics(resnet, train_data)
        logging.info(f"Epoch loss: {epoch_loss}")

        log_msg = f"Epoch {epoch}: | train_acc {tm['acc']} | train_macro_f1 {tm['macro_f1']} | train_micro_f1 {tm['micro_f1']}"
        if cfg['validate']: 
            vm = calc_metrics(resnet, valid_data)
            log_msg += f" | valid_acc {vm['acc']} | valid_macro_f1 {vm['macro_f1']} | valid_micro_f1 {vm['micro_f1']}"
        logging.info(log_msg)
        print(log_msg)

    exit()
