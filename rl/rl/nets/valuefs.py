
from torch import nn

from rl.nets.utils import net_init, fc_body


class VF(object):
    def value(self, x):
        return self.v_net(x)


class MLPVF(VF, nn.Module):
    def __init__(self, o_dim, act_type='Tanh', h_dim=(50,), device='cpu'):
        super().__init__()
        self.device = device
        self.v_net = fc_body(act_type, o_dim, h_dim)
        self.v_net.append(nn.Linear(h_dim[-1], 1))
        self.v_net = nn.Sequential(*self.v_net)
        net_init(self.v_net)
        self.to(device)
