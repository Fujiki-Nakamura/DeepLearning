import torch
from torch import nn
from .convlstm2 import ConvLSTM
from .convlstm3 import ConvLSTM as ConvLSTMTeacherForcing


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.hidden_dims = args.hidden_dims
        self.kernel_size = args.kernel_size
        self.n_layers = args.n_layers

        self.convlstm1 = ConvLSTM(
            input_size=(args.height, args.width), input_dim=args.channels,
            hidden_dim=self.hidden_dims,  kernel_size=self.kernel_size,
            num_layers=self.n_layers,
            batch_first=True, bias=True, return_all_layers=True)

    def forward(self, x):
        out, hidden_list = self.convlstm1(x)
        return out[-1], hidden_list


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.is_teacher_forcing = (
            args.teacher_forcing_ratio > 0 or args.teacher_forcing_ratio == -1)
        self.input_dim = (
            args.channels if self.is_teacher_forcing else args.hidden_dims[-1])
        self.hidden_dims = args.hidden_dims
        self.kernel_size = args.kernel_size
        self.n_layers = args.n_layers
        self.device = args.device

        if self.is_teacher_forcing or args.teacher_forcing_ratio == -1:
            # -1 for (temporal) debug
            self.convlstm1 = ConvLSTMTeacherForcing(
                input_size=(args.height, args.width), input_dim=self.input_dim,
                hidden_dim=self.hidden_dims,  kernel_size=self.kernel_size,
                num_layers=self.n_layers,
                batch_first=True, bias=True, return_all_layers=True,
                teacher_forcing_ratio=args.teacher_forcing_ratio)
        else:
            self.convlstm1 = ConvLSTM(
                input_size=(args.height, args.width), input_dim=self.input_dim,
                hidden_dim=self.hidden_dims,  kernel_size=self.kernel_size,
                num_layers=self.n_layers,
                batch_first=True, bias=True, return_all_layers=True,)

    def forward(self, x, hidden_list=None, target=None):
        out, hidden_list = self.convlstm1(x, hidden_list, target)

        return out, hidden_list


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        loss_and_reduction = args.loss.lower().split('/')
        assert len(loss_and_reduction) == 2
        self.loss, _ = loss_and_reduction
        self.logit_output = args.logit_output or self.loss == 'bce'
        self.is_teacher_forcing = (
            args.teacher_forcing_ratio > 0 or args.teacher_forcing_ratio == -1)

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, input_, target):
        out_e, hidden_e = self.encoder(input_)

        input_d = input_[:, -1, :, :, :] if self.is_teacher_forcing else out_e
        out_d, hidden_d = self.decoder(input_d, hidden_e, target)

        if self.is_teacher_forcing:
            return out_d

        # out_d: list of tensor(bs, ts, c=hidden_dim, h, w)
        out = torch.cat(out_d, dim=2)
        bs, ts, c, h, w = out.size()
        out = self.conv1x1(out.view(bs * ts, c, h, w))
        if not self.logit_output:
            out = torch.sigmoid(out)

        return out.view(bs, ts, -1, h, w)


def encdec2(args):
    return Model(args)
