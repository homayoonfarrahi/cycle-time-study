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
import torch
from torch.optim import Adam
from tqdm import tqdm

from rl.agent import Agent
from rl.buffers.buffer import Buffer
from rl.nets.policies import MLPPolicy
from rl.nets.valuefs import MLPVF
from rl.learners.ppo_test_time import PPOTestTime
import rl.envs


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

    test_batch_size = 5000
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    pol = MLPPolicy(o_dim, a_dim, h_dim=cfg['h_dim'], device=device)
    vf = MLPVF(o_dim, h_dim=cfg['h_dim'], device=device)
    np.random.set_state(random_state)
    buf = Buffer(o_dim, a_dim, test_batch_size, device=device)
    learner = PPOTestTime(pol, buf, cfg['lr'], g=cfg['g'], vf=vf, lm=cfg['lm'],
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
    for steps in range(test_batch_size):
        a, logp, dist = agent.get_action(o)
        op, r, done, infos = env.step(a)
        epi_steps += 1
        op_ = op
        agent.learner.log(o, a, r, op_, logp, dist, done)

        o = op
        ret += r
        if done:
            # tqdm.write("{} ( {} ) {:.2f}".format(steps, epi_steps, ret))
            # if not args.visualize:
            #     save_weights(pol, steps, args.weights_path)
            #     save_returns(steps, ret, args.returns_path)

            rets.append(ret)
            ret = 0
            epi_steps = 0
            o = env.reset()

    num_trials = 30
    ne_default = 10
    bs_default = 2000
    mbs_default = 50
    
    # test_ne_time(agent, bs_default, mbs_default, num_trials)
    # test_bs_time(agent, ne_default, mbs_default, num_trials)
    test_mbs_time(agent, ne_default, bs_default, num_trials)

def test_ne_time(agent, bs_default, mbs_default, num_trials):
    ne_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    for ne in ne_values:
        print('ne: ', ne)
        test_update_time(agent, ne, bs_default, mbs_default, num_trials, 'ne')

def test_bs_time(agent, ne_default, mbs_default, num_trials):
    bs_values = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    for bs in bs_values:
        print('bs: ', bs)
        test_update_time(agent, ne_default, bs, mbs_default, num_trials, 'bs')

def test_mbs_time(agent, ne_default, bs_default, num_trials):
    mbs_values = [12, 25, 37, 50, 62, 75, 87, 100, 112, 125]
    for mbs in mbs_values:
        print('mbs: ', mbs)
        test_update_time(agent, ne_default, bs_default, mbs, num_trials, 'mbs')

def test_update_time(agent, ne, bs, mbs, num_trials, prefix):
    adv_calc_times = []
    update_times = []
    x_var = {'ne': ne, 'bs': bs, 'mbs': mbs}[prefix]
    for i in range(num_trials):
        adv_calc_time, update_time = agent.learner.learn_test_time(ne, bs, mbs)
        adv_calc_times.append(adv_calc_time)
        update_times.append(update_time)

    with open('ppo_profiling/' + prefix + '_adv_times.csv', 'a', encoding='utf-8') as adv_times_file:
        adv_times_file.write(str(x_var))
        for i in range(num_trials):
            adv_times_file.write(',' + str(adv_calc_times[i]))
        adv_times_file.write('\n')

    with open('ppo_profiling/' + prefix + '_update_times.csv', 'a', encoding='utf-8') as update_times_file:
        update_times_file.write(str(x_var))
        for i in range(num_trials):
            update_times_file.write(',' + str(update_times[i]))
        update_times_file.write('\n')

def save_weights(pol, steps, path):
    torch.save(pol.state_dict(), path)
    if steps % 200000 < 150:
        torch.save(pol.state_dict(), path[:-4] + '_checkpoints/' + str(steps) + path[-4:])

def save_returns(steps, ret, path):
    with open(path, 'a', encoding='utf-8') as returns_file:
        returns_file.write(str(steps) + ',' + str(ret) + '\n')

if __name__ == "__main__":
    main()
