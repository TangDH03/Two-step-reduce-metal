import os
import os.path as path
import yaml
import numpy as np
__all__=["get_config","update_config","save_config","read_dir","add_post"]

class EasyDict(object):
    def __init__(self, opt): self.opt = opt

    def __getattribute__(self, name):
        if name == 'opt' or name.startswith("_") or name not in self.opt:
            return object.__getattribute__(self, name)
        else: return self.opt[name]

    def __setattr__(self, name, value):
        if name == 'opt': object.__setattr__(self, name, value)
        else: self.opt[name] = value
   

    def __getitem__(self, name):
        return self.opt[name]
    
    def __setitem__(self, name, value):
        self.opt[name] = value

    def __contains__(self, item):
        return item in self.opt

    def __repr__(self):
        return self.opt.__repr__()

    def keys(self):
        return self.opt.keys()

    def values(self):
        return self.opt.values()

    def items(self):
        return self.opt.items()





def get_config(config_file):
    with open(config_file) as f:
        config = (yaml.load(f,Loader=yaml.FullLoader))
    return EasyDict(config)

def update_config(config,args):
    if args is None: return
    if hasattr(args, "__dict__"): args = args.__dict__
    for arg, val in args.items():
        # if not (val is None or val is False) and arg in config: config[arg] = val
        # TODO: this may cause bugs for other programs
        if arg in config: config[arg] = val
    
    for _, val in config.items():
        if type(val) == dict: update_config(val, args)

def save_config(config, config_file, print_opts=True):
    config_str = yaml.dump(config, default_flow_style=False)
    with open(config_file, 'w') as f: f.write(config_str)
    print('================= Options =================')
    print(config_str[:-1])
    print('===========================================')

def read_dir(dir_path, predicate=None, name_only=False, recursive=False):
    if type(predicate) is str:
        if predicate in {'dir', 'file'}:
            predicate = {
                'dir': lambda x: path.isdir(path.join(dir_path, x)),
                'file':lambda x: path.isfile(path.join(dir_path, x))
            }[predicate]
        else:
            ext = predicate
            predicate = lambda x: ext in path.splitext(x)[-1]
    elif type(predicate) is list:
        exts = predicate
        predicate = lambda x: path.splitext(x)[-1][1:] in exts


def add_post(loader, post_fcn):
    class LoaderWrapper(object):
        def __init__(self, loader, post_fcn):
            self.loader = loader
            self.post_fcn = post_fcn
        
        def __getattribute__(self, name):
            if not name.startswith("__") and name not in object.__getattribute__(self, "__dict__") :
                return getattr(object.__getattribute__(self, "loader"), name)
            return object.__getattribute__(self, name)

        def __len__(self): return len(self.loader)

        def __iter__(self):
            for data in self.loader:
                yield self.post_fcn(data)

    return LoaderWrapper(loader, post_fcn)




