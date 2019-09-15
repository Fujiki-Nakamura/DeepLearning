import argparse

from experiment import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--input_reversing_ratio', type=float, default=0.)
    # network
    parser.add_argument('--model', type=str, default='convlstm1layer')
    parser.add_argument('--kernel_size', type=int, nargs='+', default=(5, 5))
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[16, ])
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5)
    parser.add_argument('--logit_output', action='store_true', default=False)
    # training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--loss', type=str, default='loss/reduction')
    # optim
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--betas', nargs='+', type=float, default=(0.9, 0.999))
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--scheduler', type=str, default='')
    parser.add_argument('--milestones', nargs='+', type=int)
    parser.add_argument('--gamma', nargs='+', type=float)
    # misc
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--expid', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true', default=False)

    args, _ = parser.parse_known_args()
    run(args)
