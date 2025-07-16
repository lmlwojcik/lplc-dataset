import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from ultralytics import YOLO
from tqdm import tqdm

from pathlib import Path
import shutil
import json
import os

from dataset.dataset_utils import LPSD_Dataset
from models.eval import calc_metrics, gen_metrics
from models.utils import find_model, start_log, end_log, log_metrics_json, dict_to_table


def train_torch_model(model, cfg, dataset, log_cfg=None):
    save_path = Path(cfg['save_path'])
    save_path.mkdir(parents=True,exist_ok=True)

    train_data = DataLoader(
        LPSD_Dataset(dataset['path'], "train", imgsz=dataset['imgsz'], device=cfg['use_gpu']),
        batch_size=cfg['batch_size'],
        shuffle=cfg['shuffle']
    )

    if cfg['validate']:
        valid_data = DataLoader(
            LPSD_Dataset(dataset['path'], "val", imgsz=dataset['imgsz'], device=cfg['use_gpu']),
            batch_size=cfg['batch_size'],
            shuffle=cfg['shuffle']
        )

    if log_cfg is not None:
        log_file = Path('logs') / Path(log_cfg['experiment_name'] + ".json")
        start_log(log_file)

    log_metrics = {}

    if cfg['use_gpu'] != -1:
        model.to(torch.device(f"cuda:{cfg['use_gpu']}"))

    if cfg['optim'] == "adam":
        opt = Adam(model.named_parameters(), **cfg['optim_config'])
    elif cfg['optim'] == "sgd":
        opt = SGD(model.named_parameters(), **cfg['optim_config'])
    loss = nn.CrossEntropyLoss()
    def train_epoch(epoch=0, step_update=-1, c_step=0):
        e_loss = 0
        pds = torch.tensor([]).to("cuda")
        gts = torch.tensor([]).to("cuda")

        with tqdm(train_data, unit="batch") as tepoch:
            for sample in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                c_step += 1

                im, lb = sample
                opt.zero_grad()

                logits = model(im)
                lb = lb.squeeze(1)
                c_loss = loss(logits, lb)

                c_loss.backward()
                opt.step()
                e_loss += c_loss.item()

                pd = logits.max(1).indices
                pds = torch.cat([pd,pds])
                gts = torch.cat([lb,gts])

                tepoch.set_postfix(loss=c_loss.item())
                if step_update != -1 and c_step % step_update == 0:
                    print(f"Epoch {epoch} at step {i}: Loss - {c_loss.item()}")

        avg_loss = e_loss/len(train_data)
        pds = pds.to(torch.int64)
        gts = gts.to(torch.int64)
        train_metrics = gen_metrics(gts, pds, pt="train", loss=avg_loss)

        return avg_loss, train_metrics

    if cfg['es_metric'].endswith("loss"):
        best_metric = 1e5
    else:
        best_metric = 0
    cnt = 0
    epoch_metrics = {}
    epoch = 0

    for epoch in range(1, cfg['epochs']+1):
        print(f"Starting epoch {epoch}")

        epoch_loss, tm = train_epoch(epoch, -1)
        log_metrics['epoch'] = epoch

        #print(tm)
        log_metrics.update(tm)
        log_metrics['train_loss'] = epoch_loss

        if cfg['validate']: 
            vm = calc_metrics(model, valid_data, "val", get_loss=True, loss=loss)
            #print(vm)
            log_metrics.update(vm)

        print(dict_to_table(log_metrics))
        #print(json.dumps(log_metrics))
        if log_cfg is not None:
            log_metrics_json(log_metrics, log_file)

        if cfg['do_es']:
            cnt += 1
            current_metric = log_metrics[cfg['es_metric']]
            if (cfg['es_metric'].endswith("loss") and current_metric < best_metric) \
                    or (current_metric > best_metric):
                best_metric = current_metric
                cnt = 0
                
                if cfg['save_best']:
                    torch.save(model, Path(cfg['save_path']) / Path("model_best.pth"))
            if cnt >= cfg['patience']:
                break
    if cfg['save_last']:
        torch.save(model, Path(cfg['save_path']) / Path(f"model_last_epoch_{epoch}.pth"))
    if epoch == 0:
        log_metrics = calc_metrics(model, train_data, "train")
        vm = calc_metrics(model, valid_data, "val")
        log_metrics.update(vm)
    if log_cfg is not None:
        log_metrics_json(log_metrics, log_file)
        end_log(log_file)

    return model, log_metrics

def train_yolo(yolo, cfg, dataset, save_dir=None):

    if save_dir is not None:
        pjdir = Path(save_dir) / Path(cfg['name'])
        #pjdir = Path(save_dir)
        cfg['project'] = Path(pjdir)

        if pjdir.exists():
            shutil.rmtree(pjdir)    
        print("Project dir:", pjdir)

    yolo.train(data=dataset['dir'], **cfg)

    return yolo, None


def test_torch_model(model, cfg, dataset, partition='test', load_model=None):
    test_data = DataLoader(
        LPSD_Dataset(dataset['path'], partition, imgsz=dataset['imgsz'], device=cfg['use_gpu']),
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

def predict_torch_model(model, cfg, dataset, partition='test', load_model=None):
    dts = LPSD_Dataset(dataset['path'], partition, imgsz=dataset['imgsz'], device=cfg['use_gpu'])
    test_data = DataLoader(
        dts,
        batch_size=1,
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

    file_predicts = []
    gts = []
    pds = []
    for f, (im,lb) in zip(dts.files, test_data):
        logits = model(im)
        lb = lb.squeeze(1)
        pd = logits.max(1).indices

        gts.append(lb)
        pds.append(pd)

        file_predicts.append({"fname": f, "gt": lb.item(), "pd": pd.item()})

    metrics = gen_metrics(torch.tensor(gts),torch.tensor(pds),pt=partition,return_matrix=True)
    ret = {'metrics': metrics, 'file_predicts': file_predicts}
    return ret

def test_yolo(model, cfg, dataset, partition='test', load_model=None):
    if model is None:
        if load_model is not None:
            model = YOLO(load_model)
        else:
            model = YOLO(f"{cfg['save_path']}/weights/best.pt")

    dts = LPSD_Dataset(dataset['dir'], partition, imgsz=cfg["imgsz"], device=-1)
    gts = []
    pds = []
    for idx,g in enumerate(dts.files):
        if not os.path.exists(g):
            continue
    
        r = model(([g]))[0]

        pd = r.probs.top1
        gt = int(g.split("/")[-2])

        gts.append(gt)
        pds.append(pd)

    return gen_metrics(torch.tensor(gts,dtype=torch.int64),
                       torch.tensor(pds,dtype=torch.int64)
                       ,pt=partition,return_matrix=True)

