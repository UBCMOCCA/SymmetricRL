import math
import torch as th
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F


class SymmetricLayer(nn.Module):
    __constants__ = ["sbias", "cbias"]
    __weights__ = ["c2s", "n2s", "c2c", "s2c", "s1s", "s2s", "n2n", "s2n"]

    def __init__(self, in_const, in_neg, in_side, out_const, out_neg, out_side, wmag=1):
        """
        """
        super().__init__()

        self.in_const = in_const
        self.in_neg = in_neg
        self.in_side = in_side
        self.out_const = out_const
        self.out_neg = out_neg
        self.out_side = out_side

        self.c2s = Parameter(th.Tensor(out_side, in_const))
        self.s1s = Parameter(th.Tensor(out_side, in_side))
        self.s2s = Parameter(th.Tensor(out_side, in_side))
        self.n2s = Parameter(th.Tensor(out_side, in_neg))
        self.sbias = Parameter(th.Tensor(out_side))

        self.s2c = Parameter(th.Tensor(out_const, in_side))
        self.c2c = Parameter(th.Tensor(out_const, in_const))
        self.cbias = Parameter(th.Tensor(out_const))

        self.n2n = Parameter(th.Tensor(out_neg, in_neg))
        self.s2n = Parameter(th.Tensor(out_neg, in_side))

        self.reset_parameters(wmag)

    def forward(self, c, n, l, r):
        c2s = F.linear(c, self.c2s, self.sbias)
        n2s = F.linear(n, self.n2s)

        if self.in_side > 0:
            s2c = F.linear((l + r) / 2, self.s2c, self.cbias)
            s2n = F.linear(l - r, self.s2n)
            s2l = F.linear(l, self.s1s) + F.linear(r, self.s2s)
            s2r = F.linear(r, self.s1s) + F.linear(l, self.s2s)
        else:
            s2c = s2n = s2l = s2r = 0

        return (
            s2c + F.linear(c, self.c2c),  # constant part
            s2n + F.linear(n, self.n2n),  # negative part
            c2s + n2s + s2l,  # left
            c2s - n2s + s2r,  # right
        )

    def reset_parameters(self, wmag):
        for wname in self.__weights__:
            init.xavier_uniform_(getattr(self, wname), gain=wmag)

        bound = 1 / math.sqrt(self.in_const + self.in_neg + 2 * self.in_side)
        self.sbias.data.fill_(0)
        self.cbias.data.fill_(0)
        # init.uniform_(self.sbias, -bound, bound)
        # init.uniform_(self.cbias, -bound, bound)


class SymmetricNetV2(nn.Module):
    def __init__(
        self,
        c_in,
        n_in,
        s_in,
        c_out,
        n_out,
        s_out,
        num_layers=3,
        hidden_size=64,
        tanh_finish=True,
    ):
        super().__init__()
        self.c_in = c_in
        self.s_in = s_in
        self.n_in = n_in
        self.c_out = c_out
        self.s_out = s_out
        self.n_out = n_out
        self.tanh_finish = tanh_finish
        assert (hidden_size % 4) == 0
        self.hidden_size = hidden_size / 4

        self.layers = []
        last_cin = c_in
        last_nin = n_in
        last_sin = s_in
        for i in range(num_layers - 1):
            self.layers.append(
                SymmetricLayer(
                    last_cin, last_nin, last_sin, hidden_size, hidden_size, hidden_size
                )
            )
            self.add_module("layer%d" % i, self.layers[i])
            last_cin, last_nin, last_sin = hidden_size, hidden_size, hidden_size
        self.layers.append(
            SymmetricLayer(last_cin, last_nin, last_sin, c_out, n_out, s_out, wmag=0.01)
        )
        self.add_module("final", self.layers[-1])

    @property
    def state_dim(self):
        return self.c_in + self.n_in + 2 * self.s_in

    @property
    def action_dim(self):
        return self.c_out + self.n_out + 2 * self.s_out

    def forward(self, obs):
        cs, ns, ss = self.c_in, self.n_in, self.s_in
        c = obs.index_select(-1, th.arange(0, cs))
        n = obs.index_select(-1, th.arange(cs, cs + ns))
        l = obs.index_select(-1, th.arange(cs + ns, cs + ns + ss))
        r = obs.index_select(-1, th.arange(cs + ns + ss, cs + ns + 2 * ss))

        for i, layer in enumerate(self.layers):
            if i != 0:
                n = th.tanh(n)  # TODO
                c = th.relu(c)
                r = th.relu(r)
                l = th.relu(l)

            c, n, l, r = layer(c, n, l, r)

        empty = th.FloatTensor(obs.shape[:-1] + (0,))
        mean = th.cat(
            [
                c if c.shape[-1] > 0 else empty,
                n if n.shape[-1] > 0 else empty,
                l if l.shape[-1] > 0 else empty,
                r if r.shape[-1] > 0 else empty,
            ],
            -1,
        )
        if self.tanh_finish:
            mean = th.tanh(mean)

        return mean

