import torch.nn
import collections

class Builder(object):
    def __init__(self, *namespaces):
        self._namespace = collections.ChainMap(*namespaces)

    def __call__(self, name, *args, **kwargs):
        try:
            return self._namespace[name](*args, **kwargs)
        except Exception as e:
            raise e.__class__(str(e), name, args, kwargs) from e

    def add_namespace(self, namespace, index=-1):
        if index >= 0:
            namespaces = self._namespace.maps
            namespaces.insert(index, namespace)
            self._namespace = collections.ChainMap(*namespaces)
        else:
            self._namespace = self._namespace.new_child(namespace)


def build_network(architecture, n_fts=None, n_cls=None, builder=Builder(torch.nn.__dict__)):
    layers = []

    print("Building network from yaml:")
    for block in architecture:
        print(block)

        assert len(block) == 1
        name, kwargs = list(block.items())[0]
        if kwargs is None:
            kwargs = {}
        args = kwargs.pop("args", [])

        if "alias" in kwargs.keys():
            alias = kwargs.pop("alias")
            if alias == "N_FT_LAYER" and n_fts is not None:
                args[1] = n_fts
            elif alias == "CLS_LAYER":
                if n_fts is not None:
                    args[0] = n_fts
                if n_cls is not None:
                    args[1] = n_cls

        layers.append(builder(name, *args, **kwargs))
    return torch.nn.Sequential(*layers)