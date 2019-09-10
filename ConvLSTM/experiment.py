import os

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from MovingMNIST import MovingMNIST
import models
from trainer import train, validate
from utils import get_logger, get_logdir, get_optimizer, save_checkpoint
from utils import get_scheduler, get_loss_fn


def run(args):
    start_epoch = 1
    best_loss = 1e+9

    # logs
    args.logdir = get_logdir(args)
    logger = get_logger(os.path.join(args.logdir, 'main.log'))
    logger.info(args)
    writer = SummaryWriter(args.logdir)

    # data
    train_set = MovingMNIST(root='./data', train=True, download=True)
    valid_set = MovingMNIST(
        root='./data', train=False, download=True, split=args.test_size)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True)
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False)

    # network
    model = models.__dict__[args.model](args=args)
    model = nn.DataParallel(model)
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(args.device)
    # training
    criterion = get_loss_fn(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best/{}'.format(args.loss)]
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info('Loaded checkpoint {} (epoch {})'.format(
                args.resume, start_epoch - 1))
        else:
            raise IOError('No such file {}'.format(args.resume))

    for epoch_i in range(start_epoch, args.epochs + 1):
        training = train(
            train_loader, model, criterion, optimizer, logger=logger, args=args)
        validation = validate(
            valid_loader, model, criterion, logger=logger, args=args)

        writer.add_scalar('Train/{}'.format(args.loss), training[args.loss], epoch_i)
        writer.add_scalar('Valid/{}'.format(args.loss), validation[args.loss], epoch_i)
        writer.add_image(
            'Image/Predict', _get_images(validation['output'], args), epoch_i)
        writer.add_image(
            'Image/Target', _get_images(validation['target'], args), epoch_i)
        message = '[{}] Epoch {} Train/{} {:.2f} Valid/{} {:.2f} '
        message = message.format(
            args.expid, epoch_i,
            args.loss, training[args.loss],
            args.loss, validation[args.loss],
        )

        is_best = validation[args.loss] < best_loss
        if is_best:
            best_loss = validation[args.loss]
            message += '(Best)'
        save_checkpoint({
            'epoch': epoch_i,
            'state_dict': model.state_dict(),
            'valid/{}'.args.loss: validation[args.loss],
            'best/{}'.args.loss: best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.logdir)

        if scheduler is not None:
            scheduler.step(epoch=epoch_i)
            logger.debug('Scheduler stepped.')
            for param_group in optimizer.param_groups:
                logger.debug(param_group['lr'])

        logger.info(message)


def _get_images(output, args):
    _ims = output.unsqueeze(2).view(-1, args.channels, args.height, args.width)
    return vutils.make_grid(_ims, nrow=10)
