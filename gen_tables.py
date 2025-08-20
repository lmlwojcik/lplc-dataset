import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

import argparse
from glob import glob
import json

def c_to_d(score, thresh=None, n_cls=4):
    if thresh is None:
        thresh = [np.mean([x/(n_cls-1),(x+1)/(n_cls-1)]) for x in range(n_cls-1)]
    idx = 0
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

def get_data(models, patterns, use_rid, do_classes, class_config=None):
    all_data = {}
    all_data_rid = {}

    for m in models:
        all_data[m] = {}
        all_data_rid[m] = {}
        for cidx,p in enumerate(patterns):
            if class_config is not None and do_classes:
                with open(class_config[cidx], "r") as fd:
                    cc = json.load(fd)
                cls = cc['class_names']
            else:
                cls = []

            fs = glob(f"saved/{m}/{p}*/**/predict*.json")
            data = {
                "train": {"overall": [], **{x:[] for x in cls}},
                "val": {"overall": [], **{x:[] for x in cls}},
                "test": {"overall": [], **{x:[] for x in cls}}
            }
            data_rid = {
                "train": {},
                "val": {},
                "test": {}
            }
            if use_rid:
                ks = list(data['train'].keys())
                for k in ks:
                    data_rid['train'][k + "_by_rid"] = []
                    data_rid['val'][k + "_by_rid"] = []
                    data_rid['test'][k + "_by_rid"] = []

            for f in fs:
                with open(f, "r") as fd:
                    res = json.load(fd)

                if len(cls) != 0:
                    n_cls = len(cls)
                else:
                    n_cls = len(np.unique([x['gt'] for x in res['file_predicts']]))

                lf = f.split("/")[-1]
                if "train" in lf:
                    tag = "train"
                elif "val" in lf:
                    tag = "val"
                elif "test" in lf:
                    tag = "test"

                if class_config is not None and do_classes:
                    gt = []
                    pd = []

                    for f in res['file_predicts']:
                        gt.append(f['gt'])
                        pd.append(f['pd'])
                    f1_overall = f1_score(gt, pd, average='micro')
                    f1_cls = f1_score(gt, pd, average=None)

                    data[tag]['overall'].append(f1_overall)
                    for i in range(len(cls)):
                        data[tag][cls[i]].append(f1_cls[i])

                    if use_rid:
                        ripd = []
                        for f in res['file_predicts']:
                            ripd.append(c_to_d(get_rid(f['logits'][0], n_cls=n_cls), n_cls=n_cls))
                        ri_f1_overall = f1_score(gt, ripd, average='micro')
                        ri_f1_cls = f1_score(gt, ripd, average=None)

                        data_rid[tag]['overall_by_rid'].append(ri_f1_overall)
                        for i in range(len(cls)):
                            data_rid[tag][f"{cls[i]}_by_rid"].append(ri_f1_cls[i])

                else:
                    f1_overall = res['metrics'][f"{tag}_micro_f1"]
                    data[tag]['overall'].append(f1_overall)

                    if use_rid:
                        ri_f1_overall = calc_metric_rid("acc", res['file_predicts'], n_cls=n_cls)
                        data_rid[tag]['overall_by_rid'].append(ri_f1_overall)
                            
            pk = p.split("_")[1] # to be changed later
            all_data[m][pk] = {"train": {}, "val": {}, "test": {}}
            all_data_rid[m][f"{pk}_rid"] = {"train": {}, "val": {}, "test": {}}
            for part in ['train', 'val', 'test']:
                for k, v in data[part].items():
                    all_data[m][pk][part][k] = np.mean(v)
                for k, v in data_rid[part].items():
                    all_data_rid[m][f"{pk}_rid"][part][k] = np.mean(v)

    if use_rid:
        for m in models:
            all_data[m].update(all_data_rid[m])

    return all_data, [f"{part}_f1" for part in list(data.keys())]

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
            cellText=[[f"{list(x.values())[0]:.4g}" for x in list(data[m][y].values())] for y in rowLabels],
            rowLabels=rowLabels,
            colLabels=metrics,
            bbox=[0, voff, hsiz, clv*(len(rowLabels)+1)]
        )
        tb.auto_set_font_size(False)
        tb.set_fontsize(12)

    plt.savefig(f"saved/figs/{out_file}.png", bbox_inches='tight', transparent=True)

def graph_classes(data, patterns, metrics, class_config, all_parts=False, out_file="classes"):
    print(data)
    models = list(data.keys())

    fig = plt.figure(num=1,clear=True)
    ax = fig.add_subplot()

    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    ax.set_frame_on(False)  # no visible frame, uncomment if size is ok

    voff = 1
    for cidx,p in enumerate(patterns):
        with open(class_config[cidx], "r") as fd:
            cc = json.load(fd)
        cls = cc['class_names']

        hsiz = 0.27*len(metrics)
        clv = 0.075
        voff -= 2*clv

        h1 = plt.table(
            cellText=[[""]],
            colLabels=[p.split("_")[1]],
            bbox=[0, voff, hsiz, clv*2]
        )
        h1.auto_set_font_size(False)
        h1.set_fontsize(12)
        voff += clv

        if all_parts:
            ###
            ### TO-DO
            ###
            rowLabels = []
        else:
            rowLabels = models
        colLabels = cls[:]
        colLabels.append("overall")
        voff -= clv*(len(rowLabels)+1)

        print([[f"{data[m][p.split('_')[1]]['test'][k]:.4g}" for k in colLabels] for m in models])

        tb = plt.table(
            cellText=[[f"{data[m][p.split('_')[1]]['test'][k]:.4g}" for k in colLabels] for m in models],
            rowLabels=rowLabels,
            colLabels=colLabels,
            bbox=[0, voff, hsiz, clv*(len(rowLabels)+1)]
        )
        tb.auto_set_font_size(False)
        tb.set_fontsize(12)


    plt.savefig(f"saved/figs/{out_file}.png", bbox_inches='tight', transparent=True)

def main(models, patterns, use_rid, do_graph_overall, do_graph_classes, class_config, file_name):
    all_data, metrics = get_data(models, patterns, use_rid, do_classes=do_graph_classes, class_config=class_config)

    if do_graph_overall:
        graph_overall(all_data, metrics, out_file=file_name)

    if do_graph_classes:
        graph_classes(all_data, patterns, metrics, class_config, out_file=file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--models", nargs='+', default=[], required=True)
    parser.add_argument('-p', '--patterns', nargs='+', default=[])

    parser.add_argument("-o", "--do_graph_overall", action='store_true')
    parser.add_argument("-c", "--do_graph_classes", action='store_true')
    parser.add_argument("-cc", "--class_config", nargs='+', default=['configs/split_configs/config_classes_base.json'])

    parser.add_argument("-r", "--use_rid", action='store_true')
    parser.add_argument("-f", "--file_name", default='graph')
    
    args = vars(parser.parse_args())

    main(**args)