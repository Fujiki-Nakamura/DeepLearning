import argparse
from collections import OrderedDict
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from MovingMNIST import MovingMNIST
import models


def main(args):
    test_set = MovingMNIST(root='./data', train=False, download=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    print('Loaded checkpoint {}'.format(args.checkpoint))
    print('Epoch {} Valid/{} {:.4f} Best/Valid/{} {:.4f}'.format(
        checkpoint['epoch'],
        args.loss, checkpoint['valid/{}'.format(args.loss)],
        args.loss, checkpoint['best/{}'.format(args.loss)]))

    new_state_dict = OrderedDict()
    for k, v in iter(checkpoint['state_dict'].items()):
        new_k = k.replace('module.', '')
        new_state_dict[new_k] = v
    model = models.__dict__[args.model](args)
    model.load_state_dict(new_state_dict)
    model.to(args.device)
    model.eval()

    inpts = np.zeros((len(test_set), 10, args.height, args.width))
    preds = np.zeros((len(test_set), 10, args.height, args.width))
    trues = np.zeros((len(test_set), 10, args.height, args.width))
    pbar = tqdm(total=len(test_loader))
    for batch_i, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.unsqueeze(2), targets.unsqueeze(2)
        inputs, targets = inputs.float() / 255., targets.float() / 255.
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        with torch.no_grad():
            outputs = model(inputs, targets)
            if args.logit_output:
                outputs = torch.sigmoid(outputs)
            outputs = outputs.squeeze(2).cpu().numpy()
            targets = targets.squeeze(2).cpu().numpy()
            inputs = inputs.squeeze(2).cpu().numpy()
            inpts[batch_i * args.batch_size:(batch_i + 1) * args.batch_size] = inputs
            preds[batch_i * args.batch_size:(batch_i + 1) * args.batch_size] = outputs
            trues[batch_i * args.batch_size:(batch_i + 1) * args.batch_size] = targets

        pbar.update(1)
    pbar.close()

    path = os.path.join(args.log_dir, 'inpts.npy')
    inpts.dump(path)
    print('Dumped at {}'.format(path))
    path = os.path.join(args.log_dir, 'preds.npy')
    preds.dump(path)
    print('Dumped at {}'.format(path))
    path = os.path.join(args.log_dir, 'trues.npy')
    trues.dump(path)
    print('Dumped at {}'.format(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--channels', type=int, default=1)
    # network
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--model', type=str, default='encdec1')
    parser.add_argument('--kernel_size', type=int, nargs='+', default=(5, 5))
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[16, ])
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.0)
    parser.add_argument('--logit_output', action='store_true', default=False)
    # training
    parser.add_argument('--loss', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    # misc
    parser.add_argument('--log_dir', type=str, default='./logs')

    args, _ = parser.parse_known_args()
    main(args)
