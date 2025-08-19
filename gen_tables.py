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

def get_data(models, patterns, use_rid):
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

    for m in models:
        all_data[m].update(all_data_rid[m])

    return all_data, list(data.keys())

def graph_overall(data, metrics, out_file="overall"):
    models = list(data.keys())

    fig = plt.figure(num=1,clear=True)
    ax = fig.add_subplot()

    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    ax.set_frame_on(False)  # no visible frame, uncomment if size is ok

    voff = 1
    for m in models:
        hsiz = 0.27*len(metrics)
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

        rowLabels = list(data[m].keys())
        voff -= clv*(len(rowLabels)+1)

        tb = plt.table(
            cellText=[[f"{x:.4g}" for x in list(data[m][x].values())] for x in rowLabels],
            rowLabels=rowLabels,
            colLabels=metrics,
            bbox=[0, voff, hsiz, clv*(len(rowLabels)+1)]
        )
        tb.auto_set_font_size(False)
        tb.set_fontsize(12)

    plt.savefig(f"saved/figs/{out_file}.png", bbox_inches='tight', transparent=True)

def graph_classes(data, metrics, class_config, out_file):
    with open(class_config, "r") as fd:
        cc = json.load(fd)
    classes = cc['class_names']

    models = list(data.keys())

    fig = plt.figure(num=1,clear=True)
    ax = fig.add_subplot()

    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    ax.set_frame_on(False)  # no visible frame, uncomment if size is ok

    voff = 1
    for m in models:
        hsiz = 0.27*len(metrics)
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

        rowLabels = list(data[m].keys())
        voff -= clv*(len(rowLabels)+1)

        tb = plt.table(
            cellText=[[f"{x:.4g}" for x in list(data[m][x].values())] for x in rowLabels],
            rowLabels=rowLabels,
            colLabels=metrics,
            bbox=[0, voff, hsiz, clv*(len(rowLabels)+1)]
        )
        tb.auto_set_font_size(False)
        tb.set_fontsize(12)

    plt.savefig(f"saved/figs/{out_file}.png", bbox_inches='tight', transparent=True)

def main(models, patterns, use_rid, do_graph_overall, do_graph_classes, class_config):
    all_data, metrics = get_data(models,patterns, use_rid)

    if do_graph_overall:
        graph_overall(all_data, metrics)

    if do_graph_classes:
        graph_classes(all_data, metrics, class_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--models", nargs='+', default=[], required=True)
    parser.add_argument('-p', '--patterns', nargs='+', default=[])

    parser.add_argument("-o", "--do_graph_overall", action='store_true')
    parser.add_argument("-c", "--do_graph_classes", action='store_true')
    parser.add_argument("-cc", "--class_config", type=str, default='configs/split_configs/config_classes_base.json')

    parser.add_argument("-r", "--use_rid", action='store_true')
    
    args = vars(parser.parse_args())

    main(**args)