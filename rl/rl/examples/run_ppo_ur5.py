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
import pybullet
import pdb

from rl.agent import Agent, UR5ReacherScriptedAgent, UR5ReacherEStopScripter
from rl.buffers.buffer import Buffer
from rl.nets.policies import MLPNormPolicy
from rl.nets.valuefs import MLPVF
from rl.learners.ppo import PPO

# from senseact.envs.ur.reacher_scripted_env import ReacherScriptedEnv
from senseact.envs.ur.reacher_env_v2 import ReacherEnvV2
from senseact.envs.ur.reacher_env_log import ReacherEnv
from senseact.utils import tf_set_seeds, NormalizedEnv


def main():
    # Setup
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-s', '--seed', required=False, type=int, default="0")
    parser.add_argument('-d', '--device', required=False)
    
    # parser.add_argument('-w', '--weights', required=False, type=str, default='weights/s_0.3_a_1.4/s_0.3_a_1.4_learn_350099.pth')
    parser.add_argument('-w', '--weights', required=False, type=str, default='weights/s_1.0_a_2.0/s_1.0_a_2.0_learn_350099.pth')
    # parser.add_argument('-w', '--weights', required=False, type=str, default='weights/s_0.3_a_1.4_v2/s_0.3_a_1.4_v2_learn_300099.pth')
    # parser.add_argument('-w', '--weights', required=False, type=str, default='weights/s_1.0_a_2.0_v2/s_1.0_a_2.0_v2_learn_300099.pth')

    parser.add_argument('-v', '--visualize', required=False, type=int, default='0')
    args = parser.parse_args()
    if args.device: device = args.device
    else: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = yaml.load(open(args.config))
    cfg['seed'] = args.seed

    # Problem
    seed = cfg['seed']
    np.random.seed(seed)
    random_state = np.random.get_state()
    torch_seed = np.random.randint(1, 2 ** 31 - 1)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)

    # env = gym.make(cfg['env_name'])
    # Create UR5 Reacher2D environment
    env = ReacherEnvV2(
            # setup="UR5_default",
            setup="UR5_2D_V2",
            host='129.128.159.210',
            dof=2,
            control_type="velocity",
            target_type="position",
            reset_type="zero",
            reward_type="precision",
            derivative_type="none",
            deriv_action_max=5,
            first_deriv_max=2,
            accel_max=1,
            speed_max=1.0,
            speedj_a=2.0,
            episode_length_time=4.0,
            episode_length_step=None,
            actuation_sync_period=1,
            dt=0.04,
            run_mode="multiprocess",
            rllab_box=False,
            movej_t=2.0,
            delay=0.0,
            random_state=random_state
        )
    env = NormalizedEnv(env)
    # Start environment processes
    env.seed(seed)
    env.start()

    # Solution
    o_dim = env.observation_space.shape[0]
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
    # agent = UR5ReacherScriptedAgent()
    # agent = UR5ReacherEStopScripter()

    if args.visualize:
        pol.load_state_dict(torch.load(args.weights))

    rets = []
    ret = 0
    epi_steps = 0
    # if args.visualize:
    #     env.render()

    o = env.reset()
    for steps in range(cfg['n_steps']):
        a, logp, dist = agent.get_action(o)
        # print('STEPPING...', steps)
        op, r, done, infos = env.step(a)
        epi_steps += 1
        op_ = op
        # if not args.visualize:
        #     agent.log_update(o, a, r, op_, logp, dist, done)

        o = op
        ret += r
        if done:
            # print('DONE')
            print(steps, "(", epi_steps, ") {0:.2f}".format(ret))
            if not args.visualize:
                # torch.save(pol.state_dict(), args.weights)
                # save_returns(steps, ret)
                # if steps % 50000 < 100:
                #     torch.save(pol.state_dict(), args.weights[:-4] + '_' + str(steps) + args.weights[-4:])
                pass

            rets.append(ret)
            ret = 0
            epi_steps = 0
            o = env.reset()
            # agent.reset()

def save_returns(steps, ret):
    with open('returns/s_0.3_a_1.4_v2_pid.csv', 'a', encoding='utf-8') as returns_file:
        returns_file.write(str(steps) + ',' + str(ret) + '\n')

if __name__ == "__main__":
    main()

