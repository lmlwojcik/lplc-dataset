import json
import random
from glob import glob
from pathlib import Path
import shutil
import argparse
import os
import copy

def save_prot_ims(pname, pt, n_classes, exclude_idx):
    path = Path('lpr_dts') / Path(f"{pname}")
    if path.exists():
        shutil.rmtree(path)
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

def save_folds(tr, vl, te, idx=0, fold_dir="folds"):
    path = Path(f"{fold_dir}")
    path.mkdir(parents=True, exist_ok=True)    

    out = {
        "train": tr,
        "val": vl,
        "test": te
    }
    with open(f"{fold_dir}/fold_{idx}.json", "w") as fd:
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

def agg_at_idxs(pls, idxs, cls, aug=None):
    if aug is not None:
        prefixes = [y.split("/")[-1].split("_")[0] for x in [v for k,v in pls.items() if k in idxs] for z in x for y in z]
    ret = {}
    for i in cls:
        ret[i] = []
        for idx in idxs:
            ret[i] += pls[int(i)][idx]
            if aug is not None:
                ret[i] += [x for x in aug[int(i)][idx] if x.split("/")[-1].split("_")[0] in prefixes]
    return ret

def gen_sym_partition(fs, sldir, fname):
    path = Path(sldir) / Path(fname)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    for i in fs.keys():
        c_path = path / Path(f"{i}")
        c_path.mkdir(parents=True, exist_ok=True)
        
        for im in fs[i]:
            src = Path(im).resolve()
            os.symlink(f"{src}", Path(c_path) / Path(os.path.basename(im)))

def gen_sym_links(tr, vl, te, fold_name, sldir):
    gen_sym_partition(tr, sldir, fold_name + "/train/")
    gen_sym_partition(vl, sldir, fold_name + "/val/")
    gen_sym_partition(te, sldir, fold_name + "/test/")

def class_mapping(c_cfg, n_classes):
    if c_cfg is None:
        return {i:i for i in range(n_classes)}
    cls = {}
    for k, v in c_cfg['class_dct'].items():
        for c in v:
            cls[int(c)] = int(k)
    return cls

def load_fnames(cfg, cls):
    plates = {int(i): [] for i in set(cls.values())}
    augmented = {int(i): [] for i in set(cls.values())}
    for i in range(cfg['n_classes']):
        if i not in cls.keys():
            continue
        if cfg['augmented_dir'] is not None:
            augmented[cls[i]] += sorted(list(glob(f"{cfg['augmented_dir']}/{i}/*")))
        plates[cls[i]] += sorted(list(glob(f"{cfg['dataset_dir']}/{i}/*")))
        if cfg['do_shuffle']:
            random.shuffle(plates[cls[i]])
    return plates, augmented

def gen_splits(cfg, c_cfg):
    nf = cfg['n_folds']
    if c_cfg is not None:
        classes = c_cfg['class_dct'].keys()
    else:
        classes = [x for x in range(len(cfg['n_classes']))]
    cls = class_mapping(c_cfg, cfg['n_classes'])
    plates, augmented = load_fnames(cfg, cls)
    for i in set(cls.values()):
        plates[i] = split_files(plates[i], nf)
    if cfg['augmented_dir'] is not None:
        for i in set(cls.values()):
            augmented[i] = split_files(augmented[i], nf)
    if c_cfg is not None:
        cfg['output_dir'] += "/" + c_cfg['sub_dir']

    n_folds = {}
    for i in range(nf):
        valid_idx = i
        valid_fs = agg_at_idxs(plates, [valid_idx], cls=classes)

        test_idxs = []
        for j in range(nf//2):
            test_idxs.append((valid_idx + j + 1) % nf)
        test_fs = agg_at_idxs(plates, test_idxs, cls=classes)

        train_idxs = []
        for j in range(nf//2):
            train_idxs.append((valid_idx - j - 1) % nf)
        train_fs = agg_at_idxs(plates, train_idxs, cls=classes, aug=augmented)

        save_folds(train_fs, valid_fs, test_fs, f"{i}_1",
                   fold_dir=cfg['output_dir'])
        n_folds[f"{i}_1"] = {"train": copy.deepcopy(train_fs),
                             "val": copy.deepcopy(valid_fs),
                             "test": copy.deepcopy(test_fs)}

        if cfg['cross_fold']:
            save_folds(test_fs, valid_fs, train_fs, f"{i}_2",
                    fold_dir=cfg['output_dir'])
            n_folds[f"{i}_2"] = {"train": copy.deepcopy(test_fs),
                                 "val": copy.deepcopy(valid_fs),
                                 "test": copy.deepcopy(train_fs)}

    return n_folds

def load_splits(cfg, c_cfg):
    fdir = Path(cfg['output_dir'])
    if c_cfg is not None:
        fdir = fdir / Path(c_cfg['sub_dir'])
    all_folds = sorted(glob(f"{fdir}/*.json"))
    ret = {}
    for f in all_folds:
        k = f.split("_")
        k = k[-2] + "_" + k[-1].split(".")[0]
        with open(f, "r") as fd:
            ret[k] = json.load(fd)
    return ret

def gen_sldirs(folds, cfg, subdir=""):
    for k,v in folds.items():
        gen_sym_links(v['train'], v['val'], v['test'], k,
                      sldir=os.path.join(cfg['sym_link_dir'], subdir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    parser.add_argument('--class_config', default=None)

    parser.add_argument('--dataset_dir', default="LPRD_Dataset/lps")
    parser.add_argument('--output_dir', default="LPRD_Dataset/folds")
    parser.add_argument('--augmented_dir', default=None, type=str)

    parser.add_argument('--do_shuffle', default=False, action='store_true')
    parser.add_argument('--folds', default=5)
    parser.add_argument('--cross_fold', default=False, action='store_true')
    parser.add_argument('--gen_sym_links', default=False, action='store_true')
    parser.add_argument('--sym_link_dir', default="sldir")

    parser.add_argument('--load_folds', action='store_true')

    args = vars(parser.parse_args())
    cfg = {}

    if args['config'] is None:
        cfg = copy.deepcopy(args)
    else:
        with open(args['config'], "r") as fd:
            cfg = json.load(fd)
        cfg['load_folds'] = args['load_folds']
    cfg['n_classes'] = 4

    if args['class_config'] is not None:
        with open(f"{args['class_config']}", "r") as fd:
            c_cfg = json.load(fd)
    else:
        c_cfg = None

    if not cfg['load_folds']:
        folds = gen_splits(cfg, c_cfg)
    else:
        folds = load_splits(cfg, c_cfg)

    if cfg['gen_sym_links']:
        if c_cfg is not None:
            gen_sldirs(folds, cfg, c_cfg['sub_dir'])
        else:
            gen_sldirs(folds, cfg)

