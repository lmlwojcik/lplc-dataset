from tqdm import tqdm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from torcheval.metrics.functional import (
        multiclass_accuracy,
        multiclass_f1_score,
        multiclass_confusion_matrix
    )

import torch

def eval_model(model, data, task, loss=None, verbose=False, device=None):
    pds = torch.tensor([]).to(device)
    gts = torch.tensor([]).to(device)

    if loss is not None:
        vloss = 0
        idx = 0
    else:
        vloss = None

    if verbose:
        data = tqdm(data)
        
    model.eval()
    with torch.no_grad():
        for sample in data:
            im, lb = sample
            logits = model(im)
            lb = lb.squeeze(1)
            
            if loss is not None:
                vloss += loss(logits, lb).item()
                idx += 1

            pd = logits.max(1).indices
                        
            pds = torch.cat([pd,pds])
            gts = torch.cat([lb,gts])

    if loss is not None:
        vloss /= idx

    return gts, pds, vloss

def gen_metrics(gts, pds, cls, pt="train", task='classification', return_matrix=False, loss=None):
    n_classes = len(cls)
    if task == 'regression':
        gts = torch.Tensor.round(gts*(n_classes-1))
        pds = torch.Tensor.round(pds*(n_classes-1))
    gts = gts.to(torch.int64)
    pds = pds.to(torch.int64)

    #micro_f1 = multiclass_f1_score(pds,gts,average='micro').item()
    macro_f1 = multiclass_f1_score(pds,gts,average='macro',num_classes=n_classes).item()
    acc = multiclass_accuracy(pds,gts).item()

    metrics = {f"{pt}_acc": acc, f"{pt}_macro_f1": macro_f1}
    #metrics = {f"{pt}_acc": acc, f"{pt}_micro_f1": micro_f1, f"{pt}_macro_f1": macro_f1}
    if loss is not None:
        metrics[f"{pt}_loss"] = loss

    if return_matrix:
        cm = multiclass_confusion_matrix(pds,gts,num_classes=n_classes).tolist()
        metrics[f"{pt}_matrix"] = cm

        df = pd.DataFrame(cm, index=cls, columns=cls).rename_axis('Ground Truth',axis='index').rename_axis('Prediction', axis='columns')
        plt.rcParams.update({'font.size': 15})
        cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
        ax = sns.heatmap(df, cmap=cmap, annot=True, fmt="d", linewidths=1, square=True)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    return metrics

def calc_metrics(
        model,
        data,
        pt="train",
        return_matrix=False,
        task='classification',
        loss=None,
        verbose=False,
        class_names=None,
        device='cpu'
    ):

    gts, pds, loss = eval_model(model, data, task,
                                loss=loss,
                                verbose=verbose, device=device)

    if class_names is None:
        class_names = ["Illegible", "Poor", "Good", "Perfect"]
    return gen_metrics(gts, pds, class_names, pt, task, return_matrix, loss=loss)
