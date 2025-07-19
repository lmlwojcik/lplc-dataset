from tqdm import tqdm

from torcheval.metrics.functional import (
        multiclass_accuracy,
        multiclass_f1_score,
        multiclass_confusion_matrix
    )

import torch

def eval_model(model, data, get_loss=False, loss=None, verbose=False, device=None):
    pds = torch.tensor([]).to(device)
    gts = torch.tensor([]).to(device)

    if get_loss:
        vloss = 0
        idx = 0
    else:
        vloss = None

    if verbose:
        data = tqdm(data)
    with torch.no_grad():
        for sample in data:
            im, lb = sample
            logits = model(im)
            lb = lb.squeeze(1)
            
            if get_loss:
                vloss += loss(logits, lb).item()
                idx += 1

            pd = logits.max(1).indices

            pds = torch.cat([pd,pds])
            gts = torch.cat([lb,gts])
    pds = pds.to(torch.int64)
    gts = gts.to(torch.int64)

    if get_loss:
        vloss /= idx

    return gts, pds, vloss

def gen_metrics(gts, pds, pt="train", return_matrix=False, loss=None, n_classes=4):
    micro_f1 = multiclass_f1_score(pds,gts,average='micro').item()
    macro_f1 = multiclass_f1_score(pds,gts,average='macro',num_classes=n_classes).item()
    acc = multiclass_accuracy(pds,gts).item()

    metrics = {f"{pt}_acc": acc, f"{pt}_micro_f1": micro_f1, f"{pt}_macro_f1": macro_f1}
    if loss is not None:
        metrics[f"{pt}_loss"] = loss

    if return_matrix:
        metrics[f"{pt}_matrix"] = multiclass_confusion_matrix(pds,gts,num_classes=4).tolist()
    return metrics

def calc_metrics(
        model,
        data,
        pt="train", return_matrix=False,
        get_loss=False, loss=None,
        verbose=False,
        n_classes=4,
        device='cpu'
    ):
    gts, pds, loss = eval_model(model, data,
                                get_loss=get_loss, loss=loss,
                                verbose=verbose, device=device)

    return gen_metrics(gts,pds,pt,return_matrix, loss=loss, n_classes=n_classes)
