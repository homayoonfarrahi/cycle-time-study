
import numpy as np

from torch import nn


def net_init(net, orth=0, w_fac=0.1, b_fac=0.0):
    if orth:
        for module in net:
            if hasattr(module, 'weight'):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if hasattr(module, 'bias'):
                nn.init.constant_(module.bias, val=0)
    else:
        net[-1].weight.data.mul_(w_fac)
        if hasattr(net[-1], 'bias'):
            net[-1].bias.data.mul_(b_fac)


def fc_body(act_type, o_dim, h_dim):
    activation = {'Tanh': nn.Tanh, 'ReLU': nn.ReLU}[act_type]
    module_list = nn.ModuleList()
    module_list.append(nn.Linear(o_dim, h_dim[0]))
    module_list.append(activation())
    for i in range(len(h_dim) - 1):
        module_list.append(nn.Linear(h_dim[i], h_dim[i + 1]))
        module_list.append(activation())
    return module_list
