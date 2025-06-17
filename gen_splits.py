import json
import random
from glob import glob
from pathlib import Path
import shutil
import argparse
import os

def save_prot_ims(pname, pt, n_classes, exclude_idx):
    path = Path('lpr_dts') / Path(f"{pname}")
    path.mkdir(parents=True, exist_ok=True)    

    for i in range(n_classes):
        if i in exclude_idx:
            continue

        c_path = path / Path(f"{i}")
        c_path.mkdir(parents=True, exist_ok=True)

        for im in pt[i]:
            shutil.copy(f"{im}", f"{c_path}/{im.split('/')[-1]}")


def save_ims(tr, vl, te, n_classes=4, exclude_idx=[]):
    save_prot_ims("train", tr, n_classes=n_classes, exclude_idx=exclude_idx)
    save_prot_ims("val", vl, n_classes=n_classes, exclude_idx=exclude_idx)
    save_prot_ims("test", te, n_classes=n_classes, exclude_idx=exclude_idx)

def save_folds(tr, vl, te, idx=0, dataset_dir="lpr_dts", fold_dir="folds"):
    path = Path(dataset_dir) / Path(f"{fold_dir}")
    path.mkdir(parents=True, exist_ok=True)    

    out = {
        "train": tr,
        "valid": vl,
        "test": te
    }
    with open(f"{dataset_dir}/{fold_dir}/fold_{idx}.json", "w") as fd:
        json.dump(out, fd, indent=2)


def split_files(files, n):
    ret = []
    for i in range(n):
        ret.append([])
    i = 0
    for f in files:
        ret[i%n].append(f)
        i += 1
    return ret

def agg_at_idxs(pls, idxs, cls=4):
    ret = {}
    for i in range(cls):
        ret[i] = []
        for idx in idxs:
            ret[i] += pls[i][idx]
    return ret

def gen_sym_partition(fs, sldir, fname, n_classes=4, exclude_idx = []):
    path = Path(sldir) / Path(fname)
    path.mkdir(parents=True, exist_ok=True)    

    for i in range(n_classes):
        if i in exclude_idx:
            continue

        c_path = path / Path(f"{i}")
        c_path.mkdir(parents=True, exist_ok=True)
        
        for im in fs[i]:
            src = Path(im).resolve()
            os.symlink(f"{src}", f"./{c_path}/{im.split('/')[-1]}")

def gen_sym_links(tr, vl, te, fold_name, sldir, exclude_idx=[]):
    gen_sym_partition(tr, sldir, fold_name + "/train/", n_classes=n_classes, exclude_idx=exclude_idx)
    gen_sym_partition(vl, sldir, fold_name + "/valid/", n_classes=n_classes, exclude_idx=exclude_idx)
    gen_sym_partition(te, sldir, fold_name + "/test/", n_classes=n_classes, exclude_idx=exclude_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', default=5)
    parser.add_argument('--dataset_dir', default="lpr_dts")
    parser.add_argument('--gen_sym_links', default=False, action='store_true')
    parser.add_argument('--sym_link_dir', default="lpr_dataset")
    parser.add_argument('--output_dir', default="folds")
    parser.add_argument('--annotation_file', default="annotations.json")

    args = vars(parser.parse_args())
    with open(f"{args['annotation_file']}", "r") as fd:
        js = json.load(fd)

    n_folds = int(args['folds'])
    n_classes = 4

    plates = {}
    for i in range(n_classes):
        plates[i] = sorted(list(glob(f"{args['dataset_dir']}/{i}/*")))
        plates[i] = split_files(plates[i], n_folds)

    for i in range(n_folds):
        valid_idx = i
        valid_fs = agg_at_idxs(plates, [valid_idx])

        test_idxs = []
        for j in range(n_folds//2):
            test_idxs.append((valid_idx + j + 1) % n_folds)
        test_fs = agg_at_idxs(plates, test_idxs)

        train_idxs = []
        for j in range(n_folds//2):
            train_idxs.append((valid_idx - j - 1) % n_folds)
        train_fs = agg_at_idxs(plates, train_idxs)

        save_folds(train_fs, valid_fs, test_fs, f"{i}_1", dataset_dir=args['dataset_dir'], fold_dir=args['output_dir'])
        save_folds(test_fs, valid_fs, train_fs, f"{i}_2", dataset_dir=args['dataset_dir'], fold_dir=args['output_dir'])

        if args['gen_sym_links']:
            gen_sym_links(train_fs, valid_fs, test_fs, f"{i}_1", sldir=args['sym_link_dir'])
            gen_sym_links(test_fs, valid_fs, train_fs, f"{i}_2", sldir=args['sym_link_dir'])

        #save_ims(train_fs, valid_fs, test_fs, exclude_idx=[3])

#    save_folds(test_fs, valid_fs, train_fs, i+5)
