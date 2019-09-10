import random
import torch
from torch import nn
from .lstms import Conv2dLSTMCell


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.device = args.device
        self.input_dim = args.channels
        self.hidden_dim = 16
        self.h = args.height
        self.w = args.width
        self.convlstm1 = Conv2dLSTMCell(
            args.input_dim, args.hidden_dim,
            kernel_size=(args.kernel_size, args.kernel_size),
            stride=args.stride, padding=args.kernel_size // 2)

    def forward(self, input_):
        bs, ts, h, w = input_.size()
        h_0, c_0 = self.init_hidden(bs)
        for ti in range(ts):
            h_1, (h_1, c_1) = self.convlstm1(
                input_[:, ti, :, :], (h_0, c_0))
            h_0, c_0 = h_1, c_1
        out = None  # TODO?

        return out, (h_1, c_1)

    def init_hidden(self, batch_size):
        size = (batch_size, self.hidden_dim, self.h, self.w)
        return (
            torch.zeros(*size, device=self.device),
            torch.zeros(*size, device=self.device)
        )


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.device = args.device
        self.input_dim = args.channels
        self.hidden_dim = 16
        self.teacher_forcing_ratio = args.teacher_forcing_ratio

        self.convlstm1 = Conv2dLSTMCell(
            args.input_dim, args.hidden_dim,
            kernel_size=(args.kernel_size, args.kernel_size),
            stride=args.stride, padding=args.kernel_size // 2)
        self.conv1x1 = nn.Conv2d(
            self.hidden_dim, self.input_dim, kernel_size=(1, 1), stride=1)
        self.sigmoid = nn.Sigmoid()

        random.seed(args.random_seed)

    def forward(self, input_, target, h_0_and_c_0):
        h_0, c_0 = h_0_and_c_0
        bs, ts, h, w = target.size()
        input_current = input_[:, -1, :, :]
        output = torch.zeros(bs, ts, h, w)
        for ti in range(ts):
            h_1, (h_1, c_1) = self.convlstm1(input_current, (h_0, c_0))
            out = self.sigmoid(self.conv1x1(h_1))
            output[:, ti, :, :] = out
            if random.random() < self.teacher_forcing_ratio:
                input_current = target[:, ti, :, :]
            else:
                input_current = out

        return output


class ConvLSTM(nn.Module):
    def __init__(self, args):
        super(ConvLSTM, self).__init__()
        self.encoder = Encoder(args)
        self.decocer = Decoder(args)

    def forward(self, input_, target):
        o_enc, h_and_c_enc = self.encoder(input_)
        out = self.decoder(input_, target, h_and_c_enc)
        return out


def convlstm1layer(args):
    return ConvLSTM(args)
