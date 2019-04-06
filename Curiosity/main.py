import argparse
import warnings ; warnings.filterwarnings('ignore') 
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.distributions.categorical import Categorical

import cv2
import numpy as np 
from model import ActorCriticNetwork
from scipy.misc import imresize 
from PIL import Image
import matplotlib.pyplot as plt
from env import env
from agent import Agent

from saliency_helper import *

parser = argparse.ArgumentParser(description='Baseline')

parser.add_argument('--lr', type = float, default = 0.0003, metavar = 'LR',
                    help = 'Learning rate (Default: 0.0003)')
parser.add_argument('--gamma', type = float, default = 0.99, metavar = 'G',
                    help = 'Discount Factor for Rewards (Default: 0.99)')
parser.add_argument('--use_gae', type = bool, default = True, metavar = 'GAE',
                    help = 'Use GAE? (Default: False)')
parser.add_argument('--tau', type = float, default = 0.95, metavar = 'T',
                    help = 'Parameter for GAE (Default: 0.95)')
parser.add_argument('--num_steps', type = int, default = 128, metavar = 'NS',
                    help = 'Number of steps in a episode (Default: 20)')
parser.add_argument('--epoch', type = int, default = 3, metavar = 'EPOCH',
                    help = 'number of epochs? (Default : 3)')
parser.add_argument('--batch_size', type = int, default = 32 , metavar = 'BS',
                    help = 'Batch Size? (Default : 32)')
parser.add_argument('--ppo_eps', type = float, default = 0.1 , metavar = 'PPO',
                    help = 'Parameter for ppo clipping (Default : 0.1)')
parser.add_argument('--is_render', type = bool, default = True , metavar = 'RENDER',
                    help = 'render the environment? (Default : False)')
parser.add_argument('--training', type = bool, default = True , metavar = 'TR',
                    help = 'Train the model? (Default : True)')
parser.add_argument('--load_model', type = bool, default = False , metavar = 'TR',
                    help = 'Load a pre trained model? (Default : False)')
parser.add_argument('--use_cuda', type = bool, default = True, metavar = 'CUDA',
                    help = 'Using GPU for faster Processing (Default : True)')
### For final Project, not required for baseline.                  
parser.add_argument('--env_id', default = 'BreakoutDeterministic-v4', metavar = 'ENV',
                    help = 'environment to train on (default: BreakoutDeterministic-v4)')
parser.add_argument('--reward_type', default = 'Dense', metavar = 'RW',
                    help = 'Type of reward setting (Dense or Sparse, default: Dense)')
parser.add_argument('--icm', default = 'False', metavar = 'ICM',
                    help = 'Intrinsic Reward for agent (default: False)')



if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device('cuda' if args.use_cuda else 'cpu')
    env_ob = env(args)
    # print(env_ob.env.observation_space.shape, env_ob.env.action_space.n)
    agent = Agent(args,env_ob.env.observation_space.shape, env_ob.env.action_space)
    states = np.zeros([1, 4, 84, 84])
    if args.load_model:
        agent.model.load_state_dict(torch.load('my-BreakoutDeterministic-v4.model'))
    

    reward_rms = agent.RunningMeanStd()
    obs_rms = agent.RunningMeanStd(shape=(1, 1, 84, 84))

    pre_obs_norm_step = 10000
    global_step = 0
    frames = []
    frames_rgb = []
    next_obs = []
    all_i = []
    d = False
    while True:
        all_state, all_reward, all_done, all_next_state, all_action = [], [], [], [], []
        global_step += args.num_steps

        # for i in range(args.num_steps):
        next_states, rewards, dones = [], [], []

        s, r, d, actions,s_rgb = env_ob.run(agent, states)
        next_obs.append(s[3, :, :].reshape([1, 84, 84]))

        next_states.append(s)
        rewards.append(r)
        dones.append(d)

        next_obs = np.stack(next_obs)
        obs_rms.update(next_obs)

        frames.append(s)
        frames_rgb.append(s_rgb)
        next_states = np.array(next_states)
        rewards = np.hstack(rewards)
        dones = np.hstack(dones)

        intrinsic_reward = agent.compute_intrinsic_reward(
                (states - obs_rms.mean) / np.sqrt(obs_rms.var),
                (next_states - obs_rms.mean) / np.sqrt(obs_rms.var),
                actions)

        intrinsic_reward = np.hstack(intrinsic_reward)
        all_i += intrinsic_reward
        all_state.append(states)
        all_next_state.append(next_states)
        all_reward.append(rewards)
        all_done.append(dones)
        all_action.append(actions)
        
        states = next_states[:, :, :, :]
        
        
        if args.training:
            all_state = np.stack(all_state).transpose(
                [1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
            all_next_state = np.stack(all_next_state).transpose(
                [1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
            all_reward = np.stack(all_reward).transpose().reshape([-1])
            all_action = np.stack(all_action).transpose().reshape([-1])
            all_done = np.stack(all_done).transpose().reshape([-1])

            value, next_value, policy = agent.state_transition(all_state, all_next_state)

            all_target, all_adv = [], []

            target, adv = agent.get_advantage(all_reward, all_done, value, next_value)
            all_target.append(target)
            all_adv.append(adv)

            #agent.train(all_state, np.hstack(all_target), all_action, np.hstack(all_adv))

            if global_step % (args.num_steps * 100) == 0:
                torch.save(agent.model.state_dict(), 'my-BreakoutDeterministic-v4.model')
    
    