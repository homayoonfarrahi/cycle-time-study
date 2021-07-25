import sys
import os
import math
import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
# from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from utils import hard_update

# import pybullet_env_mods

from senseact.envs.ur.reacher_env_v2 import ReacherEnvV2
# from senseact.envs.ur.reacher_env_log import ReacherEnv
from senseact.utils import tf_set_seeds, NormalizedEnv

def main():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default="HighFreqReacherBulletEnv-v0",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=50000, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=5000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--act_dt', required=False, type=float, default='0.04')
    parser.add_argument('--update_period', required=False, type=float, default='1.0')
    parser.add_argument('--g_exp', required=False, type=float, default='1.0')
    parser.add_argument('--sim_dt', required=False, type=float, default='0.04')
    args = parser.parse_args()

    # dt-aware
    # args.gamma = args.gamma ** (args.act_dt / 0.04)
    args.gamma = args.gamma ** (args.g_exp)
    args.replay_size = int(args.replay_size * (0.04 / args.act_dt))

    env_dir_id = 'urreacher'
    weights_dir = 'sac_runs/dt_weights_' + env_dir_id + '/' + str(args.seed)
    returns_dir = 'sac_runs/dt_returns_' + env_dir_id + '/' + str(args.seed)
    # run_id = str(args.act_dt) + '_' + str(args.batch_size) + '_' + str(args.update_period) + '_' + str(args.sim_dt)
    run_id = '{}_{}_{}_{}_{}_{}_{}_{}'.format(args.act_dt, args.batch_size, args.update_period, args.g_exp, args.replay_size, args.tau, args.target_update_interval, args.sim_dt)
    weights_path_actor = weights_dir + '/' + run_id + '_actor.pth'
    weights_path_critic = weights_dir + '/' + run_id + '_critic.pth'
    returns_path = returns_dir + '/' + run_id + '.csv'
    if os.path.exists(returns_path):
        print('This run has happened before. Returns are available at: ' + returns_path)
        sys.exit()

    for dirname in [weights_dir, returns_dir]:
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

    args.num_steps = args.num_steps * int(0.04 // args.sim_dt)
    args.start_steps = args.start_steps * int(0.04 // args.sim_dt)

    # Environment
    # env = NormalizedActions(gym.make(args.env_name))
    # env = gym.make(args.env_name)
    # env.seed(args.seed)
    # env.action_space.seed(args.seed)

    seed = args.seed
    np.random.seed(seed)
    random_state = np.random.get_state()
    torch_seed = np.random.randint(1, 2 ** 31 - 1)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)

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
            dt=args.sim_dt,
            run_mode="multiprocess",
            rllab_box=False,
            movej_t=2.0,
            delay=0.0,
            random_state=random_state
        )
    env = NormalizedEnv(env)
    # Start environment processes
    env.seed(seed)
    env.action_space.seed(seed)
    env.start()


    # NOTE there should be a better way to do this
    # env.robot.clip_scale = 1.0

    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)

    # Agent
    updated_agent = SAC(env.observation_space.shape[0], env.action_space, args)
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    hard_update(agent.policy, updated_agent.policy)

    #Tesnorboard
    # writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
    #                                                              args.policy, "autotune" if args.automatic_entropy_tuning else ""))

    # Memory
    memory = ReplayMemory(args.replay_size, seed)

    # Training Loop
    total_numsteps = 0
    updates = 0
    total_agent_steps = 0

    # ac_interval = args.act_dt // args.sim_dt
    ac_interval = float(math.floor(args.act_dt / args.sim_dt))
    for i_episode in itertools.count(1):
        r_buffer = 0
        agent_steps = 0
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            if episode_steps % ac_interval == 0:
                if len(memory) > args.batch_size:
                    if True: #(args.update_period <= 1) or ((args.update_period > 1) and (total_agent_steps % args.update_period == 0)):
                        hard_update(agent.policy, updated_agent.policy)

                if args.start_steps > total_numsteps:
                    action = env.action_space.sample()  # Sample random action
                else:
                    action = agent.select_action(state)  # Sample action from policy
                act_state = state

                if len(memory) > args.batch_size:
                    if (args.update_period <= 1) or ((args.update_period > 1) and (total_agent_steps % args.update_period == 0)):
                        epochs = 1
                        if args.update_period <= 1:
                            epochs = int(1.0 / args.update_period)

                        # Number of updates per step in environment
                        for i in range(epochs):
                            # Update parameters of all the networks
                            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = updated_agent.update_parameters(memory, args.batch_size, updates)

                            # writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                            # writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                            # writer.add_scalar('loss/policy', policy_loss, updates)
                            # writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                            # writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                            updates += 1

            next_state, reward, done, _ = env.step(action) # Step
            r_buffer += reward

            next_state_ = next_state
            if (episode_steps + 1) % ac_interval == 0 or done:
                # Ignore the "done" signal if it comes from hitting the time horizon.
                # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                mask = 1 if episode_steps + 1 == env._episode_length_step else float(not done)

                memory.push(act_state, action, r_buffer, next_state_, mask) # Append transition to memory
                r_buffer = 0
                agent_steps += 1
                total_agent_steps += 1

            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            state = next_state

        # writer.add_scalar('reward/train', episode_reward, i_episode)
        print("Episode: {}, total numsteps: {}, agent steps: {}, reward: {}".format(i_episode, total_numsteps, agent_steps, round(episode_reward, 2)))
        updated_agent.save_model(env_name='urreacher', actor_path=weights_path_actor, critic_path=weights_path_critic)
        save_returns(total_numsteps, episode_reward, agent_steps, returns_path)

        if total_numsteps > args.num_steps:
            break

        if i_episode % 10 == 0 and args.eval is True:
            avg_reward = 0.
            episodes = 10
            for _  in range(episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                while not done:
                    action = agent.select_action(state, evaluate=True)

                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward


                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes


            # writer.add_scalar('avg_reward/test', avg_reward, i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")

    env.close()

def save_returns(steps, ret, agent_steps, path):
    with open(path, 'a', encoding='utf-8') as returns_file:
        returns_file.write('{},{},{}\n'.format(steps, ret, agent_steps))

if __name__ == "__main__":
    main()
