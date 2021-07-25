
import torch
from torch import nn
from torch.distributions import Normal, Categorical

from rl.nets.utils import net_init, fc_body


class Policy(object):
    def action(self, x):
        """
        :param x: tensor of shape [N, 1], where N is number of observations
        :return:
            action: of shape [N, 1]
            lprob: of shape [N, 1]
        """
        with torch.no_grad():
            dist = self.dist(x)
            action = dist.sample()
            lprob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, lprob, dist

    def logp_dist(self, x, a):
        pass

    def dist(self, x):
        pass

    def dist_to(self, dist, to_device='cpu'):
        pass

    def dist_stack(self, dists):
        pass

    def dist_index(self, dist, ind):
        pass


class MLPNormPolicy(Policy, nn.Module):
    def __init__(self, o_dim, a_dim, act_type='Tanh', h_dim=(50,), log_std=0, std_scale=1.0, device='cpu'):
        super().__init__()
        self.device = device
        mean_net = fc_body(act_type, o_dim, h_dim)
        mean_net.append(nn.Linear(h_dim[-1], a_dim))
        self.mean_net = nn.Sequential(*mean_net)
        net_init(self.mean_net)
        self.log_std = nn.Parameter(torch.ones(a_dim) * log_std)
        self.std_scale = std_scale
        self.to(device)

    def logp_dist(self, x, a):
        dist = self.dist(x)
        lprob = dist.log_prob(torch.as_tensor(a, device=self.device)).sum(1, keepdim=True)
        return lprob, dist

    def dist(self, x):
        x = x.to(self.device)
        action_mean = self.mean_net(x)
        return Normal(action_mean, torch.exp(self.log_std) * self.std_scale)

    def dist_to(self, dist, to_device='cpu'):
        dist.loc.to(to_device)
        dist.scale.to(to_device)
        return dist

    def dist_stack(self, dists, device='cpu'):
        return Normal(
            torch.cat(tuple([dists[i].loc for i in range(len(dists))])).to(device),
            torch.cat(tuple([dists[i].scale for i in range(len(dists))])).to(device)
        )

    def dist_index(self, dist, ind):
        return Normal(dist.loc[ind], dist.scale[ind])

class MLPCatPolicy(Policy, nn.Module):
    def __init__(self, o_dim, num_acs, act_type='Tanh', h_dim=(50,), device='cpu'):
        super().__init__()
        self.device = device
        ac_prefs_net = fc_body(act_type, o_dim, h_dim)
        ac_prefs_net.append(nn.Linear(h_dim[-1], num_acs))
        net_init(ac_prefs_net)
        self.ac_prefs_net = nn.Sequential(*ac_prefs_net)
        self.to(device)

    def logp_dist(self, x, a):
        dist = self.dist(x)
        lprob = dist.log_prob(torch.as_tensor(a, device=self.device).squeeze()).unsqueeze(1)
        return lprob, dist

    def dist(self, x):
        x = x.to(self.device)
        action_prefs = self.ac_prefs_net(x)
        return Categorical(logits=action_prefs)

    def dist_to(self, dist, to_device='cpu'):
        dist.logits.to(to_device)
        dist.probs.to(to_device)
        return dist

    def dist_stack(self, dists, device='cpu'):
        return Categorical(logits=torch.cat(tuple([dists[i].logits for i in range(len(dists))])).to(device))

    def dist_index(self, dist, ind):
        return Categorical(logits=dist.logits[ind])
