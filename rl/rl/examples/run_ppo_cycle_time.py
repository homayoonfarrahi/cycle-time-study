import sys
import os
import yaml
import argparse
import numpy as np
# import matplotlib as mpl
# mpl.use("TKAgg")

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
from rl.nets.policies import MLPNormPolicy
from rl.nets.valuefs import MLPVF
from rl.learners.ppo import PPO
import pybullet_env_mods


def main():
    # Setup
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-s', '--seed', required=False, type=int, default="0")
    parser.add_argument('-d', '--device', required=False)
    parser.add_argument('-w', '--weights_path', required=False, type=str, default='dt_weights/0/0.016_10_2000_50_0.008.pth')
    parser.add_argument('-r', '--returns_path', required=False, type=str, default='dt_returns/0/0.016_10_2000_50_0.008.csv')
    parser.add_argument('-v', '--visualize', required=False, type=int, default='0')
    # parser.add_argument('--n_steps', required=False, type=int, default='1250000')
    parser.add_argument('--n_steps', required=False, type=int, default='2500000')
    parser.add_argument('--n_itrs', required=False, type=int, default='10')
    parser.add_argument('--bs', required=False, type=int, default='2000')
    parser.add_argument('--mbs', required=False, type=int, default='50')
    parser.add_argument('--act_dt', required=False, type=float, default='0.016')
    parser.add_argument('--g_exp', required=False, type=float, default='1.0')
    parser.add_argument('--lm_exp', required=False, type=float, default='1.0')
    args = parser.parse_args()

    if args.device: device = args.device
    else: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = yaml.load(open(args.config))
    cfg['seed'] = args.seed

    # ep_len = 2.4
    # sim_dt = 0.002

    ep_len = 16.0
    sim_dt = 0.004

    # run_id = str(args.act_dt) + '_' + str(args.n_itrs) + '_' + str(args.bs) + '_' + str(args.mbs) + '_' + str(args.g_exp) + '_' + str(args.lm_exp) + '_' + str(sim_dt)
    run_id = str(args.act_dt) + '_' + str(args.n_itrs) + '_' + str(args.bs) + '_' + str(args.mbs) + '_' + str(sim_dt)
    returns_dir = 'ppo_runs/dt_returns_noscale/' + str(cfg['seed'])
    args.weights_path = 'ppo_runs/dt_weights_invdblpndlm_default/' + str(cfg['seed']) + '/' + run_id + '.pth'
    args.returns_path = returns_dir + '/' + run_id + '.csv'

    g = cfg['g']
    lm = cfg['lm']
    # g = cfg['g'] ** (args.act_dt / 0.016)
    # lm = cfg['lm'] ** (args.act_dt / 0.016)
    # g = min(cfg['g'], cfg['g'] ** (args.act_dt / 0.016))
    # lm = min(cfg['lm'], cfg['lm'] ** (args.act_dt / 0.016))
    # g = cfg['g'] ** (args.g_exp)
    # lm = cfg['lm'] ** (args.lm_exp)
    n_steps = args.n_steps * int(0.016 // sim_dt)
    exploration_scale = 1.0# np.sqrt(0.016 / args.act_dt)

    if os.path.exists(args.returns_path):
        print('This run has happened before. Returns are available at: ' + args.returns_path)
        sys.exit()

    if not args.visualize:
        dirnames = [returns_dir, args.weights_path[:-4] + '_checkpoints']
        for dirname in dirnames:
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)

    # Problem
    seed = cfg['seed']
    env = gym.make(cfg['env_name'])
    env.seed(seed)

    # NOTE there should be a better way to do this
    env.robot.clip_scale = exploration_scale

    # Solution
    np.random.seed(seed)
    random_state = np.random.get_state()
    torch_seed = np.random.randint(1, 2 ** 31 - 1)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)

    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    pol = MLPNormPolicy(o_dim, a_dim, h_dim=cfg['h_dim'], std_scale=exploration_scale, device=device)
    vf = MLPVF(o_dim, h_dim=cfg['h_dim'], device=device)
    np.random.set_state(random_state)
    buf = Buffer(o_dim, a_dim, args.bs, device=device)
    learner = PPO(pol, buf, cfg['lr'], g=g, vf=vf, lm=lm,
                   OptPol=Adam, OptVF=Adam,
                   u_epi_up=cfg['u_epi_ups'], device=device,
                   n_itrs=args.n_itrs, n_slices=args.bs // args.mbs,
                   u_adv_scl=cfg['u_adv_scl'],
                   clip_eps=cfg['clip_eps'],
                   u_joint_opt=cfg['u_joint_opt'], max_grad_norm=cfg['max_grad_norm']
                  )
    agent = Agent(pol, learner, device=device)
    if args.visualize:
        pol.load_state_dict(torch.load(args.weights_path))

    rets = []
    ret = 0
    ret_components = np.zeros(3)
    epi_steps = 0
    ac_interval = args.act_dt // sim_dt
    r_buffer = 0
    agent_steps = 0
    if args.visualize:
        env.render()

    o = env.reset()
    for steps in tqdm(range(n_steps)):
        if epi_steps % ac_interval == 0:
            a, logp, dist = agent.get_action(o)
            act_o = o

        op, r, done, infos = env.step(a)
        r_buffer += r
        op_ = op
        if (epi_steps + 1) % ac_interval == 0 or done:
            if not args.visualize:
                agent.log_update(act_o, a, r_buffer, op_, logp, dist, done)
            r_buffer = 0
            agent_steps += 1

        epi_steps += 1

        o = op
        ret += r
        # ret_components += infos['reward_components']
        if done:
            tqdm.write("{} ( {} ) {:.2f}".format(steps, agent_steps, ret))
            if not args.visualize:
                save_weights(pol, steps, args.weights_path, n_steps, ep_len, sim_dt)
                save_returns(steps, ret, ret_components, agent_steps, args.returns_path)

            rets.append(ret)
            ret = 0
            ret_components = np.zeros(3)
            r_buffer = 0
            epi_steps = 0
            agent_steps = 0
            o = env.reset()

def save_weights(pol, steps, path, n_steps, ep_len, sim_dt):
    torch.save(pol.state_dict(), path)
    if steps % (n_steps / 10) < ep_len / sim_dt:
        torch.save(pol.state_dict(), path[:-4] + '_checkpoints/' + str(steps) + path[-4:])

def save_returns(steps, ret, ret_components, agent_steps, path):
    with open(path, 'a', encoding='utf-8') as returns_file:
        returns_file.write('{},{},{},{},{},{}\n'.format(
                                                        steps,
                                                        ret,
                                                        agent_steps,
                                                        ret_components[0],
                                                        ret_components[1],
                                                        ret_components[2]
                                                 )
        )

if __name__ == "__main__":
    main()
