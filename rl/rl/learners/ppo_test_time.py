import time
import numpy as np
import torch
from torch import nn

from rl.learners.ppo import PPO


class PPOTestTime(PPO):
    def learn_test_time(self, ne, bs, mbs):
        os, acts, rs, op, logpbs, _, dones = self.buf.get(self.pol.dist_stack)
        os, acts, rs, op, logpbs, dones = os[:bs], acts[:bs], rs[:bs], op[:bs], logpbs[:bs], dones[:bs]

        with torch.no_grad():
            pre_vals = self.vf.value(torch.cat((os, op)))

        adv_calc_start_time = time.time()
        v_rets, advs = self.get_rets_advs(rs, dones, pre_vals.t()[0])
        adv_calc_time = time.time() - adv_calc_start_time

        inds = np.arange(os.shape[0])
        update_start_time = time.time()
        for itr in range(ne):
            np.random.shuffle(inds)
            for start in range(0, len(os), mbs):
                ind = inds[start:start + mbs]
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

        update_time = time.time() - update_start_time

        return adv_calc_time, update_time
