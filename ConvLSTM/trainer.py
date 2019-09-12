import torch
from tqdm import tqdm
from utils import AverageMeter


def train(dataloader, model, criterion, optimizer, logger=None, args=None):
    model.train()
    losses = AverageMeter()
    pbar = tqdm(total=len(dataloader))
    for i, (input_, target) in enumerate(dataloader):
        bs, ts, h, w = target.size()
        output, _target, loss = step(
            input_, target, model, criterion, args=args)
        n_elements = _get_n_elements(bs, ts, h, w, args)
        losses.update(loss.item(), n_elements)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.update(1)
        if args.debug: break  # noqa
    pbar.close()

    if args.loss.lower().startswith('bce'):
        output = torch.sigmoid(output)
    logger.debug(
        'output/min {} output/max {} target/min {} target/max {}'.format(
            output.min().item(), output.max().item(),
            _target.min().item(), _target.max().item()))

    return {args.loss: losses.avg, 'output': output.cpu(), 'target': _target.cpu()}


def validate(dataloader, model, criterion, logger=None, args=None):
    model.eval()
    losses = AverageMeter()
    pbar = tqdm(total=(len(dataloader)))
    for i, (input_, target) in enumerate(dataloader):
        bs, ts, h, w = target.size()
        with torch.no_grad():
            output, _target, loss = step(
                input_, target, model, criterion, args=args)
        n_elements = _get_n_elements(bs, ts, h, w, args)
        losses.update(loss.item(), n_elements)

        pbar.update(1)
        if args.debug: break  # noqa
    pbar.close()

    if args.loss.lower().startswith('bce'):
        output = torch.sigmoid(output)

    return {args.loss: losses.avg, 'output': output.cpu(), 'target': _target.cpu()}


def step(input_, target, model, criterion, args):
    bs, ts, h, w = target.size()
    input_ = input_.float() / 255.
    target = target.float() / 255.
    input_, target = input_.to(args.device), target.to(args.device)
    output = model(input_.unsqueeze(2), target.unsqueeze(2))

    # (bs, ts, c, h, w) -> (bs, ts, h, w) -> (ts, bs, h, w)
    output = output.squeeze(2).permute(1, 0, 2, 3)
    # (bs, ts, h, w) -> (ts, bs, h, w)
    target = target.permute(1, 0, 2, 3)

    assert len(output) == len(target) == ts
    loss = 0.
    reduction = args.loss.split('/')[-1].lower()
    for t_i in range(ts):
        if reduction == 'image':
            loss += criterion(output[t_i], target[t_i]) / bs / ts
        elif args.reduction.lower().startswith('mean'):
            loss += criterion(output[t_i], target[t_i])

    # output, target returned in batch_first shape
    return output.permute(1, 0, 2, 3), target.permute(1, 0, 2, 3), loss


def _get_n_elements(bs, ts, h, w, args):
    reduction = args.loss.split('/')[-1].lower()
    if reduction == 'image':
        n = bs * ts
    elif args.reduction.lower().startswith('mean'):
        n = bs * h * w
    return n
