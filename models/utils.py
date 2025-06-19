
def dict_to_string(dct):
    # For loging purposes
    ret = ""
    for k, v in dct.items():
        ret += f"| {k}: {v}"
    return ret