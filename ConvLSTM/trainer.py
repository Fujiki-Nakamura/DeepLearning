import torch
from tqdm import tqdm
from utils import AverageMeter


def train(dataloader, model, criterion, optimizer, logger=None, args=None):
    model.train()
    losses = AverageMeter()
    pbar = tqdm(total=len(dataloader))
    for i, (input_, target) in enumerate(dataloader):
        bs, ts, h, w = target.size()
        output, loss = step(
            input_, target, model, criterion, args=args)
        n = bs * h * w if args.reduction.lower().startswith('mean') else bs
        losses.update(loss.item(), n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.update(1)
        if args.debug: break  # noqa
    pbar.close()

    return {args.loss: losses.avg}


def validate(dataloader, model, criterion, logger=None, args=None):
    model.eval()
    losses = AverageMeter()
    pbar = tqdm(total=(len(dataloader)))
    for i, (input_, target) in enumerate(dataloader):
        bs, ts, h, w = target.size()
        with torch.no_grad():
            outputs, loss = step(
                input_, target, model, criterion, args=args)
        n = bs * h * w if args.reduction.lower().startswith('mean') else bs
        losses.update(loss.item(), n)

        pbar.update(1)
        if args.debug: break  # noqa
    pbar.close()

    return {args.loss: losses.avg}


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
    for t_i in range(ts):
        if args.reduction.lower().startswith('mean'):
            loss += criterion(output[t_i], target[t_i])
        else:
            loss += criterion(output[t_i], target[t_i]) / bs

    return output, loss
