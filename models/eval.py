from torcheval.metrics.functional import (
        multiclass_accuracy,
        multiclass_f1_score,
        multiclass_confusion_matrix
    )

import torch

def eval_model(model, data):
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

    return gts, pds

def gen_metrics(gts, pds, pt="train", return_matrix=False):
    micro_f1 = multiclass_f1_score(pds,gts,average='micro').item()
    macro_f1 = multiclass_f1_score(pds,gts,average='macro',num_classes=4).item()
    acc = multiclass_accuracy(pds,gts).item()

    metrics = {f"{pt}_acc": acc, f"{pt}_micro_f1": micro_f1, f"{pt}_macro_f1": macro_f1}

    if return_matrix:
        metrics[f"{pt}_matrix"] = multiclass_confusion_matrix(pds,gts,num_classes=4).tolist()
    return metrics

def calc_metrics(model, data, pt="train", return_matrix=False):
    gts, pds = eval_model(model, data)
    return gen_metrics(gts,pds,pt,return_matrix)


