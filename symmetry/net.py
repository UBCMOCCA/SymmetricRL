import torch as th
import torch.nn as nn


class SymmetricNet(nn.Module):
    def __init__(self, net, c_out, n_out, s_out):
        assert net.state_dim % 2 == 0
        half = net.state_dim / 2

        super().__init__()
        self.net = net
        self.state_dim = net.state_dim
        self.action_dim = net.action_dim

        # the buffers are automatically transferred between devices
        self.register_buffer(
            "mirror_inds",
            th.cat([th.arange(half, net.state_dim), th.arange(0, half)]).long(),
        )
        self.register_buffer("c_range", th.arange(0, c_out))
        self.register_buffer("n_range", th.arange(c_out, c_out + n_out))
        self.register_buffer("s_range", th.arange(c_out + n_out, c_out + n_out + s_out))

    def forward(self, obs):
        mobs = obs.index_select(-1, self._buffers["mirror_inds"])
        out_o = self.net(obs)
        out_m = self.net(mobs)
        # original
        c_o = out_o.index_select(-1, self._buffers["c_range"])
        n_o = out_o.index_select(-1, self._buffers["n_range"])
        s_o = out_o.index_select(-1, self._buffers["s_range"])
        # mirrored
        c_m = out_m.index_select(-1, self._buffers["c_range"])
        n_m = out_m.index_select(-1, self._buffers["n_range"])
        s_m = out_m.index_select(-1, self._buffers["s_range"])

        return th.cat(
            [
                # commons
                (c_o + c_m) / 2,
                # opposites/negatives
                (n_o - n_m) / 2,
                # side 1
                s_o,
                # side 2
                s_m,
            ],
            -1,
        )


class SymmetricVNet(nn.Module):
    def __init__(self, net, state_dim):
        assert state_dim % 2 == 0
        half = state_dim / 2
        super().__init__()
        self.net = net
        self.register_buffer(
            "mirror_inds",
            th.cat([th.arange(half, state_dim), th.arange(0, half)]).long(),
        )

    def forward(self, obs):
        mobs = obs.index_select(-1, self._buffers["mirror_inds"])
        return (self.net(obs) + self.net(mobs)) / 2
