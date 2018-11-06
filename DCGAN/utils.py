# coding: UTF-8
import os


def save_args(log_dir, args):
    fpath = os.path.join(log_dir, 'args.txt')
    with open(fpath, 'w') as f:
        f.writelines(
            ['{}: {}\n'.format(arg, getattr(args, arg))
             for arg in dir(args) if not arg.startswith('_')])
    os.system('cat {}'.format(fpath))
