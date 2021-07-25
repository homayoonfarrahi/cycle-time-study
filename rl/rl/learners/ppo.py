
import numpy as np
import torch
from torch import nn

from rl.learners.learner import Learner


class PPO(Learner):
    """
    Implementation of PPO
    """
    def __init__(self, pol, buf, lr, g, vf, lm,
                 OptPol, OptVF,
                 device='cpu',
                 u_epi_up=0,  # whether update epidocially or not
                 n_itrs=10,
                 n_slices=32,
                 u_adv_scl=1,  # scale return with mean and std
                 clip_eps=0.2,
                 u_joint_opt=0,  # use a single optimizer jointly for both actor and critic
                 max_grad_norm=1000000000,  # maximum gradient norm for clip_grad_norm
                 ):
        self.pol = pol
        self.buf = buf
        self.g = g
        self.vf = vf
        self.lm = lm
        if u_joint_opt:
            self.opt_pol = OptPol(list(self.pol.parameters()) + list(self.vf.parameters()), lr=lr)
        else:
            self.opt_pol = OptPol(self.pol.parameters(), lr=lr)
            self.opt_vf = OptVF(self.vf.parameters(), lr=lr)
        self.device =device
        self.u_epi_up = u_epi_up
        self.n_itrs = n_itrs
        self.n_slices = n_slices
        self.u_adv_scl = u_adv_scl
        self.clip_eps = clip_eps
        self.u_joint_opt = u_joint_opt
        self.max_grad_norm = max_grad_norm

    def log(self, o, a, r, op, logpb, dist, done):
        self.buf.store(o, a, r, op, logpb, dist, done)

    def learn_time(self, done):
        return (not self.u_epi_up or done) and len(self.buf.o_buf) >= self.buf.bs

    def post_learn(self):
        self.buf.clear()

    def get_rets_advs(self, rs, dones, vals, device='cpu'):
        dones, rs, vals = dones.to(device), rs.to(device), vals.to(device)
        advs = torch.as_tensor(np.zeros(len(rs)+1, dtype=np.float32), device=device)
        for t in reversed(range(len(rs))):
            delta = rs[t] + (1-dones[t])*self.g*vals[t+1] - vals[t]
            advs[t] = delta + (1-dones[t])*self.g*self.lm*advs[t+1]
        v_rets = advs[:-1] + vals[:-1]
        advs = advs[:-1].view(-1, 1)
        if self.u_adv_scl:
            advs = advs - advs.mean()
            if advs.std() != 0 and not torch.isnan(advs.std()): advs /= advs.std()
        v_rets, advs = v_rets.to(self.device), advs.to(self.device)
        return v_rets.view(-1, 1), advs

    def learn(self):
        os, acts, rs, op, logpbs, distbs, dones = self.buf.get(self.pol.dist_stack)
        with torch.no_grad():
            pre_vals = self.vf.value(torch.cat((os, op)))
        v_rets, advs = self.get_rets_advs(rs, dones, pre_vals.t()[0])
        inds = np.arange(os.shape[0])
        mini_bs = self.buf.bs // self.n_slices
        for itr in range(self.n_itrs):
            np.random.shuffle(inds)
            for start in range(0, len(os), mini_bs):
                ind = inds[start:start + mini_bs]
                # Policy update preparation
                logpts, dist = self.pol.logp_dist(os[ind], acts[ind])
                grad_sub = (logpts - logpbs[ind]).exp()
                p_loss0 = - (grad_sub * advs[ind])
                ext_loss = - (torch.clamp(grad_sub, 1 - self.clip_eps, 1 + self.clip_eps) * advs[ind])
                p_loss = torch.max(p_loss0, ext_loss)
                p_loss = p_loss.mean()
                # value update preparation
                vals = self.vf.value(os[ind])
                v_loss = ((v_rets[ind] - vals).pow(2)).mean()

                # Policy update
                if self.u_joint_opt:
                    p_loss += v_loss
                self.opt_pol.zero_grad()
                p_loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(list(self.pol.parameters()) + list(self.vf.parameters()), self.max_grad_norm)
                self.opt_pol.step()

                # Value update
                if not self.u_joint_opt:
                    self.opt_vf.zero_grad()
                    v_loss.backward()
                    if self.max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(self.vf.parameters(), self.max_grad_norm)
                    self.opt_vf.step()
        return {}
