import datetime as dt
import os
import shutil
import torch
from torch import nn, optim


def get_loss_fn(args):
    if args.loss.lower().startswith('bce'):
        loss_fn = nn.BCEWithLogitsLoss(reduction=args.reduction)
    elif args.loss.lower().startswith('mse'):
        loss_fn = nn.MSELoss(reduction=args.reduction)
    elif args.loss.lower().startswith('l1'):
        loss_fn = nn.L1Loss(reduction=args.reduction)
    else:
        raise NotImplementedError

    return loss_fn


def get_optimizer(model, args):
    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
    elif args.optim.lower() == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=args.lr, alpha=0.99, weight_decay=args.weight_decay)
    return optimizer


def get_scheduler(optimizer, args):
    if args.scheduler.lower() == 'multisteplr':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, args.milestones, args.gamma)
    else:
        return None
    return scheduler


def get_logdir(args):
    if args.expid == '':
        args.expid = dt.datetime.now().strftime('%Y%m%d%H%M%S')
    logdir = os.path.join(args.logdir, args.expid)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    os.chmod(logdir, 0o0777)
    return logdir


def get_logger(log_file):
    from logging import getLogger, FileHandler, StreamHandler
    from logging import Formatter, DEBUG, INFO
    fh = FileHandler(log_file)
    fh.setLevel(DEBUG)
    sh = StreamHandler()
    sh.setLevel(INFO)
    for handler in [fh, sh]:
        formatter = Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
    logger = getLogger('log')
    logger.setLevel(DEBUG)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def save_checkpoint(state, is_best, logdir):
    filename = os.path.join(logdir, 'checkpoint.pt')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(logdir, 'best.pt'))


class AverageMeter(object):
    """Computes and stores the average and current value
        adopted from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L296
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
