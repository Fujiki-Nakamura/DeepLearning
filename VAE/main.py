import argparse

from experiment import Experiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--log_freq', type=int, default=10000)
    args = None

    Experiment(args).run()
