import os
import yaml
import argparse
import numpy as np
import matplotlib as mpl
mpl.use("TKAgg")

try:
    import pybullet_envs
except ImportError:
    print("pybullet_envs not available")
import gym
from gym.spaces import Discrete
import torch
from torch.optim import Adam
from tqdm import tqdm
import pybullet

from rl.agent import Agent
from rl.buffers.buffer import Buffer
from rl.nets.policies import MLPNormPolicy, MLPCatPolicy
from rl.nets.valuefs import MLPVF
from rl.learners.ppo import PPO
import pybullet_env_mods


def main():
    # Setup
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-s', '--seed', required=False, type=int, default="0")
    parser.add_argument('-d', '--device', required=False)
    parser.add_argument('-w', '--weights_path', required=False, type=str, default='dt_weights/3/0.0165.pth')
    parser.add_argument('-r', '--returns_path', required=False, type=str, default='dt_returns/3/0.0165.csv')
    parser.add_argument('-v', '--visualize', required=False, type=int, default='0')
    args = parser.parse_args()
    if args.device: device = args.device
    else: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = yaml.load(open(args.config))
    cfg['seed'] = args.seed
    if not args.visualize:
        dirname = args.weights_path[:-4] + '_checkpoints'
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    # Problem
    seed = cfg['seed']
    env = gym.make(cfg['env_name'])
    env.seed(seed)

    # Solution
    np.random.seed(seed)
    random_state = np.random.get_state()
    torch_seed = np.random.randint(1, 2 ** 31 - 1)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)

    o_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, Discrete):
        a_dim = 1
        num_acs = env.action_space.n
        pol = MLPCatPolicy(o_dim, num_acs, h_dim=cfg['h_dim'], device=device)
    else:
        a_dim = env.action_space.shape[0]
        pol = MLPNormPolicy(o_dim, a_dim, h_dim=cfg['h_dim'], device=device)
    
    vf = MLPVF(o_dim, h_dim=cfg['h_dim'], device=device)
    np.random.set_state(random_state)
    buf = Buffer(o_dim, a_dim, cfg['bs'], device=device)
    learner = PPO(pol, buf, cfg['lr'], g=cfg['g'], vf=vf, lm=cfg['lm'],
                   OptPol=Adam, OptVF=Adam,
                   u_epi_up=cfg['u_epi_ups'], device=device,
                   n_itrs=cfg['n_itrs'], n_slices=cfg['n_slices'],
                   u_adv_scl=cfg['u_adv_scl'],
                   clip_eps=cfg['clip_eps'],
                   u_joint_opt=cfg['u_joint_opt'], max_grad_norm=cfg['max_grad_norm']
                  )
    agent = Agent(pol, learner, device=device)
    if args.visualize:
        pol.load_state_dict(torch.load(args.weights_path))

    rets = []
    ret = 0
    epi_steps = 0
    if args.visualize:
        env.render()

    o = env.reset()
    for steps in tqdm(range(cfg['n_steps'])):
        a, logp, dist = agent.get_action(o)
        op, r, done, infos = env.step(a)
        epi_steps += 1
        op_ = op
        if not args.visualize:
            agent.log_update(o, a, r, op_, logp, dist, done)

        o = op
        ret += r
        if done:
            tqdm.write("{} ( {} ) {:.2f}".format(steps, epi_steps, ret))
            if not args.visualize:
                save_weights(pol, steps, args.weights_path)
                save_returns(steps, ret, args.returns_path)

            rets.append(ret)
            ret = 0
            epi_steps = 0
            o = env.reset()

def save_weights(pol, steps, path):
    torch.save(pol.state_dict(), path)
    if steps % 200000 < 150:
        torch.save(pol.state_dict(), path[:-4] + '_checkpoints/' + str(steps) + path[-4:])

def save_returns(steps, ret, path):
    with open(path, 'a', encoding='utf-8') as returns_file:
        returns_file.write(str(steps) + ',' + str(ret) + '\n')

if __name__ == "__main__":
    main()
