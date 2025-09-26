import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from itertools import product
import matplotlib as mpl

from glob import glob
import argparse
import json

def metrics_from_matrix(mat, cls, cls_dct):
    metrics = {}
    gt = []
    pd = []
    if cls_dct is not None:
        dct = {v:int(k) for k,vv in cls_dct.items() for v in vv}
        for i in range(len(mat[0])):
            if i not in dct.keys():
                continue

            gt += [dct[i]]*sum(mat[i])
            for j in range(len(mat[0])):
                if j not in dct.keys():
                    pd += [-1]*mat[i][j]
                else:
                    pd += [dct[j]]*mat[i][j]
    else:
        for i in range(len(cls)):
            gt += [i]*sum(mat[i])
            for j in range(len(cls)):
                pd += [j]*mat[i][j]
    report = classification_report(gt, pd, output_dict=True)

    metrics['overall'] = report['weighted avg']['f1-score']
    for i,c in enumerate(cls):
        metrics[c] = report[str(i)]['f1-score']

    return metrics

def get_overalls(m, e, cls, cls_dct, subdir, metric):
    overalls = {}
    for o in ['train', 'val', 'test']:
        overalls[o] = []
    overalls.update({c: [] for c in cls})
    e_p = glob(f"saved/{m}/{e}_*")

    for p in e_p:
        if subdir in p:
            cls_arg = None
        else:
            cls_arg = cls_dct

        rep = {}
        with open(f"{p}/all_results.json", "r") as fd:
            js = json.load(fd)
            rep = metrics_from_matrix(js['test_matrix'], cls, cls_arg)
            rep['train'] = js[f"train_{metric}"]
            rep['val'] = js[f"val_{metric}"]
            rep['test'] = js[f"test_{metric}"]
        for c in overalls.keys():
            overalls[c].append(rep[c])

    for c in overalls.keys():
        overalls[c] = np.mean(overalls[c])
    return overalls

def gen_table(results, output, title):
    ms = list(results.keys())
    ex = list(results[ms[0]].keys())
    metrics = list(results[ms[0]][ex[0]].keys())

    data = np.zeros((len(ms), len(ex), len(metrics)))
    for i,m in enumerate(ms):
        for j,e in enumerate(ex):
            for k,p in enumerate(metrics):
                data[i,j,k] = results[m][e][p]
    first_rows = list(product(ms, ex))

    # Arguments for table
    voff = 1
    #hsiz = 2
    hsiz = 0.27*(len(metrics)+2) # Total table width
    clv = 0.085                  # cell height
    #voff -= (len(first_rows)+1)*clv # controls vertical delta

    # if title is not None:
    #     plt.title(title)
    fig = plt.figure(figsize=(hsiz*6,clv*(len(first_rows)+1)*6))
    #fig = plt.figure(figsize=(hsiz*6,clv*(len(first_rows)+1)*6))
    print(hsiz, clv*(len(first_rows)+1))
    plt.axis('off')
    plt.grid('off')

    h1 = plt.table(
        cellText=[list(first_rows[i]) + [f"{x:.4g}" for x in data[i//len(ex),i%len(ex)]]
                   for i in range(len(first_rows))],
        colLabels=["Model", "Scenario"] + metrics,
        colWidths= [(0.35*len(x))/(len(ms[0]) + len(ex[0])) for x in [ms[0], ex[0]]] + [0.65/len(metrics)]*len(metrics),
        bbox=[0,0,1,1],
        #bbox=[0, 0.5-clv*(len(first_rows)+1)/2, hsiz, clv*(len(first_rows)+1)]
    )
    h1.set_fontsize(12)
    h1.auto_set_font_size(False)


    h1.figure.savefig(f"saved/figs/{output}.png", transparent=False, bbox_inches='tight')

def main(models, experiments, class_config, transform, output, metric, title):
    with open(f"configs/split_configs/{class_config}", "r") as fd:
        cls_cfg = json.load(fd)
        subdir = cls_cfg['sub_dir']
        cl = cls_cfg['class_names']
        cls_dct = cls_cfg['class_dct'] if transform else None

    rs = {}
    for m in models:
        rs[m] = {}
        for i,e in enumerate(experiments):
            results = get_overalls(m, e, cl, cls_dct, subdir, metric)
            rs[m][e] = results

    gen_table(rs, output, title)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--models", nargs='+', default=[], required=True)
    parser.add_argument('-x', '--experiments', nargs='+', default=[])

    parser.add_argument("-c", "--class_config", default=['base.json'])
    parser.add_argument('-t', '--transform', action='store_true')

    parser.add_argument("-mt", "--metric", default='acc')
    parser.add_argument("-tl", "--title", default=None)

    parser.add_argument("-o", "--output", default='report')
    
    args = vars(parser.parse_args())

    main(**args)