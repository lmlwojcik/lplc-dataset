from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score
import torch

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


