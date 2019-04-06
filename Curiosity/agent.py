import gym
import os
import random

import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import cv2

from model import *

import torch.optim as optim
from torch.distributions.categorical import Categorical


class Agent():
    def __init__(self, args, input_size, output_size):
        self.model = ActorCriticNetwork(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.num_steps = args.num_steps
        self.gamma = args.gamma
        self.tau = args.tau
        self.use_gae = args.use_gae
        self.use_cuda = args.use_cuda
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.ppo_eps = args.ppo_eps
        self.clip_grad_norm = 0.5
        self.icm = ICMModel(input_size, output_size)

        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = self.model.to(self.device)

        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.icm.parameters()), lr = self.learning_rate)
        self.icm = self.icm.to(self.device)


    def get_action(self, state):
        state = torch.Tensor(state).float().to(self.device)
        policy, value = self.model(state)
        policy = F.softmax(policy, dim= -1).data.cpu().numpy()
        r = np.expand_dims(np.random.rand(policy.shape[0]), axis=1)
        action = (policy.cumsum(axis=1) > r).argmax(axis=1)

        return action

    def state_transition(self, state, next_state):
        state = torch.from_numpy(state).float().to(self.device)
        policy, value = self.model(state)

        next_state = torch.from_numpy(next_state).float().to(self.device)
        _, next_value = self.model(next_state)

        value = value.data.cpu().numpy().squeeze()
        next_value = next_value.data.cpu().numpy().squeeze()

        return value, next_value, policy

    def compute_intrinsic_reward(self, state, next_state, action):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)

        action_onehot = torch.FloatTensor(
            len(action), self.output_size).to(
            self.device)
        action_onehot.zero_()
        action_onehot.scatter_(1, action.view(len(action), -1), 1)

        real_next_state_feature, pred_next_state_feature, pred_action = self.icm(
            [state, next_state, action_onehot])
        intrinsic_reward = self.eta * (real_next_state_feature - pred_next_state_feature).pow(2).sum(1) / 2

        return intrinsic_reward.data.cpu().numpy()

    def train(self, states, next_states, target, actions, advantage, old_policy):
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        target = torch.FloatTensor(target).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        advantage = torch.FloatTensor(advantage).to(self.device)

        sample_range = np.arange(len(states))
        ce = nn.CrossEntropyLoss()
        forward_mse = nn.MSELoss()

        with torch.no_grad():
            old_policy, old_value = self.model(states)
            m_old = Categorical(F.softmax(old_policy, dim = -1))
            old_log_prob = m_old.log_prob(actions)

        for i in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(states)/ self.batch_size)):
                sample_id = sample_range[self.batch_size*j : self.batch_size * (j+1)]

                action_onehot = torch.FloatTensor(self.batch_size, self.output_size).to(self.device)
                action_onehot.zero_()
                action_onehot.scatter_(1, actions[sample_id].view(-1,1),1)
                real_next_state_feature, pred_next_state_feature, pred_action = self.icm(\
                    states[sample_id], next_states[sample_id], action_onehot)

                inverse_loss = ce(pred_action, actions[sample_id])

                forward_loss = forward_mse(pred_next_state_feature, real_next_state_feature.detach())


                policy, value = self.model(states[sample_id])
                m = Categorical(F.softmax(policy, dim = -1))
                log_prob = m.log_prob(actions[sample_id])

                ratio = torch.exp(log_prob - old_log_prob[sample_id])

                surr1 = ratio * advantage[sample_id]
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps) * advantage[sample_id]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(value.sum(1), target[sample_id])

                entropy = m.entropy().mean()
                
                self.optimizer.zero_grad()
                # loss = actor_loss + critic_loss
                loss = (actor_loss + 0.5 * critic_loss - 0.001 * entropy) + forward_loss + inverse_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()


    def get_advantage(self, reward, done, value, next_value):
        discounted_return = np.zeros([self.num_steps])

        if self.use_gae:
            gae = 0
            for t in range(self.num_steps -1, -1, -1):
                delta = reward[t] + self.gamma * next_value[t] * (1 - done[t]) - value[t]
                gae = delta + self.gamma * self.tau * (1 - done[t]) * gae
                discounted_return[t] = gae + value[t]

            advantage = discounted_return - value

        else:

            dr = next_value[-1]
            for t in range(self.num_steps -1 , -1 , -1):
                dr = reward[t] + self.gamma * dr * (1 - done[t])
                discounted_return[t] = dr
            advantage = discounted_return - value

        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-30)

        return discounted_return, advantage

    class RunningMeanStd(object):
        def __init__(self, epsilon=1e-4, shape=()):
            self.mean = np.zeros(shape, 'float64')
            self.var = np.ones(shape, 'float64')
            self.count = epsilon

        def update(self, x):
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

        def update_from_moments(self, batch_mean, batch_var, batch_count):
            delta = batch_mean - self.mean
            tot_count = self.count + batch_count

            new_mean = self.mean + delta * batch_count / tot_count
            m_a = self.var * (self.count)
            m_b = batch_var * (batch_count)
            M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
            new_var = M2 / (self.count + batch_count)

            new_count = batch_count + self.count

            self.mean = new_mean
            self.var = new_var
            self.count = new_count



    

