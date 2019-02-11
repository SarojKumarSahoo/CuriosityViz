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



parser = argparse.ArgumentParser(description='Baseline')

parser.add_argument('--lr', type = float, default = 0.003, metavar = 'LR',
                    help = 'Learning rate (Default: 0.0001)')
parser.add_argument('--gamma', type = float, default = 0.99, metavar = 'G',
                    help = 'Discount Factor for Rewards (Default: 0.99)')
parser.add_argument('--tau', type = float, default = 1.00, metavar = 'T',
                    help = 'Parameter for GAE (Default: 1.00)')
parser.add_argument('--steps', type = int, default = 20, metavar = 'NS',
                    help = 'Number of steps in a episode (Default: 20)')
parser.add_argument('--render', default = 'False', metavar = 'RENDER',
                    help = 'render the environment? (Default : False)')
parser.add_argument('--use_cuda', default = 'False', metavar = 'CUDA',
                    help = 'Using GPU for faster Processing (Default : False)')
### For final Project, not required for baseline.                  
parser.add_argument('--env', default = 'BreakoutDeterministic-v4', metavar = 'ENV',
                    help = 'environment to train on (default: BreakoutDeterministic-v4)')
parser.add_argument('--reward_type', default = 'Dense', metavar = 'RW',
                    help = 'Type of reward setting (Dense or Sparse, default: Dense)')
parser.add_argument('--icm', default = 'False', metavar = 'ICM',
                    help = 'Intrinsic Reward for agent (default: False)')



if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device('cuda' if args.use_cuda else 'cpu')

    env = gym.make(args.env)
    # test = imresize(env.reset()[35:195].mean(2), (80,80))
    # plt.imshow(test)
    # plt.show()
    preprocess = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255
    model = ActorCriticNetwork(1, env.action_space)

    state = preprocess(env.reset())
    state = torch.from_numpy(state)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    done = True
    print ("state: ", state.shape)
    env.render()

    while True:
        if done:
            cx = Variable(torch.zeros(1, model.lstm_size))
            hx = Variable(torch.zeros(1, model.lstm_size))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        
        values = []
        log_probs = []
        actions = []
        rewards = []
        entropies = []

        for step in range(10):#range(args.steps):
            value, policy, (hx, cx) = model(
                (Variable(state.view(1,1,80,80)), (hx, cx)))
            
            policy = F.softmax(policy)
            entropy = Categorical(F.softmax(policy, dim=-1)).entropy()
            entropies.append(entropy)

            action = policy.multinomial(num_samples=1).data
            actions.append(action)
            log_prob = F.log_softmax(policy).gather(1, Variable(action))
            env.render()
            next_state, reward, done, _ = env.step(action.numpy())
            done = done 
            reward = max(min(reward, 1), -1)

            if done:
                episode_length = 0
                state = preprocess(env.reset())

            state = torch.from_numpy(preprocess(next_state))
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        if not done:
            value, policy, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data

        values.append(Variable(R))
        R = Variable(R)
        total_target = []
        gae = 0
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]

            delta_t = rewards[i] + args.gamma * values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t
            total_target.append(R)
            
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        m = Categorical(F.softmax(policy, dim=-1))
        mse = nn.MSELoss()
        actor_loss = -m.log_prob(torch.FloatTensor(actions)) * gae

        entropy = m.entropy()

        critic_loss = mse(value.sum(1), total_target)

        loss = actor_loss.mean() + 0.5 * critic_loss -  entropy.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    env.close()
