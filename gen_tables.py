import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score

import argparse
from glob import glob
import json

def c_to_d(score, thresh):
    for idx, v in enumerate(thresh):
        if score < v:
            return idx
    return idx+1

def get_rid(logits, n_cls):
    logits = torch.Tensor(logits).softmax(dim=0)
    return sum([x*(i/(n_cls-1)) for i,x in enumerate(logits)])

def calc_metric_rid(metric, logits, n_cls=4):
    clt = [np.mean([x/(n_cls-1),(x+1)/(n_cls-1)]) for x in range(n_cls-1)]
    gt = []
    pd = []

    for v in logits:
        pd.append(c_to_d(get_rid(v['logits'][0], n_cls), clt))
        gt.append(v['gt'])

    if "macro_f1" in metric:
        return f1_score(gt, pd, average="macro")
    elif "acc" in metric:
        return f1_score(gt, pd, average="micro")

def main(models, patterns, use_rid):
    all_data = {}
    all_data_rid = {}
    for m in models:
        all_data[m] = {}
        all_data_rid[m] = {}
        for p in patterns:
            fs = glob(f"saved/{m}/{p}*/**/all_results.json")
            data = {
                'train_acc': [],
                'val_acc': [],
                'test_acc': [],
                'train_macro_f1': [],
                'val_macro_f1': [],
                'test_macro_f1': []
            }
            data_rid = {}
            if use_rid:
                ks = list(data.keys())
                for k in ks:
                    data_rid[k + "_by_rid"] = []

            for f in fs:
                with open(f, "r") as fd:
                    res = json.load(fd)
                if use_rid:
                    with open(f.replace("all_results", "predict_results_test_with_logits"), "r") as fd:
                        with_logits = json.load(fd)
                    n_cls = len(np.unique([x['gt'] for x in with_logits['file_predicts']]))

                for k, v in data.items():
                    v.append(res[k])
                for k, v in data_rid.items():
                    v.append(calc_metric_rid(k, with_logits['file_predicts'], n_cls=n_cls))
            
            pk = p.split("_")[1]
            all_data[m][pk] = {}
            all_data_rid[m][f"{pk}_rid"] = {}
            for k, v in data.items():
                all_data[m][pk][k] = np.mean(v)
            for k, v in data_rid.items():
                all_data_rid[m][f"{pk}_rid"][k] = np.mean(v)

    fig, ax = plt.subplots() # set size frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    ax.set_frame_on(False)  # no visible frame, uncomment if size is ok

    voff = 1
    for m in models:
        hsiz = 0.27*len(data.keys())
        all_data[m].update(all_data_rid[m])
        clv = 0.075
        voff -= 2*clv
        h1 = plt.table(
            cellText=[[""]],
            colLabels=[m],
            bbox=[0, voff, hsiz, clv*2]
        )
        h1.auto_set_font_size(False)
        h1.set_fontsize(12)
        voff += clv

        rowLabels = list(all_data[m].keys())
        voff -= clv*(len(rowLabels)+1)

        tb = plt.table(
            cellText=[[f"{x:.2g}" for x in list(all_data[m][x].values())] for x in rowLabels],
            rowLabels=rowLabels,
            colLabels=list(data.keys()),
            bbox=[0, voff, hsiz, clv*(len(rowLabels)+1)]
        )
        tb.auto_set_font_size(False)
        tb.set_fontsize(12)

    plt.savefig("ab.png", bbox_inches='tight', transparent=True)

    print(all_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--models", nargs='+', default=[], required=True)
    parser.add_argument("-r", "--use_rid", action='store_true')
    parser.add_argument('-p', '--patterns', nargs='+', default=[])
    
    args = vars(parser.parse_args())

    main(**args)