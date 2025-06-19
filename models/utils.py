from natsort import natsorted
from glob import glob

def dict_to_string(dct):
    # For logging purposes
    ret = ""
    for k, v in dct.items():
        ret += f"| {k}: {v}"
    return ret


def find_model(save_path):
    # By the syntax the outputted models, the last one in the list is the latest
    models = natsorted(list(glob(f"{save_path}/*.pth")))

    for m in models:
        if "best" in m:
            return m
    chosen = m
    return chosen