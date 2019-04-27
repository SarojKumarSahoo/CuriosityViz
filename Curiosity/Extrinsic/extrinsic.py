import gym
import os
import random
from itertools import chain

import numpy as np
from sklearn.manifold import TSNE

import torch.nn.functional as F
import torch.nn as nn
import torch
import cv2

import logging 
import csv

from model import *

import torch.optim as optim
from torch.multiprocessing import Pipe, Process

from collections import deque
from sklearn.utils import shuffle

from tensorboardX import SummaryWriter


class AtariEnvironment(Process):
    def __init__(
            self,
            env_id,
            is_render,
            env_idx,
            child_conn,
            history_size=4,
            h=84,
            w=84):
        super(AtariEnvironment, self).__init__()
        self.daemon = True
        self.env = gym.make(env_id)
        self.env_id = env_id
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn
        self.actions_list = []
        self.actions_map = ['noop', 'fire', 'right', 'left']
        self.total_actions_list = []
        self.rewards_list = []
        self.total_rewards_list = []
        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.episode_data = []
        self.r_all_list = []
        self.h = h
        self.w = w

        self.reset()
        self.lives = self.env.env.ale.lives()

    def run(self):
        super(AtariEnvironment, self).run()
        while True:
            life_done = True
            action= self.child_conn.recv()
            if self.is_render:
                self.env.render()

            if 'Breakout' in self.env_id:
                action += 1
            _, reward, done, info = self.env.step(action)

            if self.env_idx == 1: 
                self.actions_list.append(self.actions_map[action])
                self.rewards_list.append(reward)
            if life_done:
                if self.lives > info['ale.lives'] and info['ale.lives'] > 0:
                    force_done = True
                    self.lives = info['ale.lives']
                else:
                    force_done = done
            else:
                force_done = done

            self.history[:3, :, :] = self.history[1:, :, :]
            self.history[3, :, :] = self.pre_proc(
                self.env.env.ale.getScreenGrayscale().squeeze().astype('float32'))

            self.rall += reward
            self.steps += 1

            if done:
                self.recent_rlist.append(self.rall)
                print("[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}".format(
                    self.episode, self.env_idx, self.steps, self.rall, np.mean(self.recent_rlist)))
                if self.env_idx == 1:
                    self.total_actions_list.append(self.actions_list)
                    self.actions_list= []

                    self.r_all_list.append(np.sum(np.array(self.rewards_list)))

                    self.total_rewards_list.append(self.rewards_list)
                    self.rewards_list = []

                    self.episode_data.append([self.episode, self.steps, np.mean(self.r_all_list)])

                    with open("actions_1.csv", 'w', newline='') as resultFile:
                        wr = csv.writer(resultFile)
                        wr.writerows(self.total_actions_list)
                    with open("rewards_1.csv", 'w', newline='') as resultFile:
                        wr = csv.writer(resultFile)
                        wr.writerows(self.total_rewards_list)
                    with open("episode_data.csv", 'w', newline='') as resultFile:
                        wr = csv.writer(resultFile)
                        wr.writerows(self.episode_data)
                self.history = self.reset()

            self.child_conn.send(
                [self.history[:, :, :], reward, force_done, done])

    def reset(self):
        self.steps = 0
        self.episode += 1
        self.rall = 0
        self.env.reset()
        self.lives = self.env.env.ale.lives()
        self.get_init_state(
            self.env.env.ale.getScreenGrayscale().squeeze().astype('float32'))
        return self.history[:, :, :]

    def pre_proc(self, X):
        x = cv2.resize(X, (self.h, self.w))
        x *= (1.0 / 255.0)

        return x

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)


class ActorAgent(object):
    def __init__(
            self,
            input_size,
            output_size,
            num_env,
            num_step,
            gamma,
            lam=0.95,
            use_gae=True,
            use_cuda=False,
            use_noisy_net=False):
        self.model = CnnActorCriticNetwork(
            input_size, output_size, use_noisy_net)
        self.num_env = num_env
        self.output_size = output_size
        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma
        self.lam = lam
        self.use_gae = use_gae

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.model = self.model.to(self.device)

    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        state = state.float()
        policy, value, f = self.model(state)
        policy = F.softmax(policy, dim=-1).data.cpu().numpy()
        print(policy)
        action = self.random_choice_prob_index(policy)

        return action,f

    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    def forward_transition(self, state, next_state):
        state = torch.from_numpy(state).to(self.device)
        state = state.float()
        policy, value, _ = self.model(state)
        
        next_state = torch.from_numpy(next_state).to(self.device)
        next_state = next_state.float()
        _, next_value, _ = self.model(next_state)

        value = value.data.cpu().numpy().squeeze()
        next_value = next_value.data.cpu().numpy().squeeze()

        return value, next_value, policy

    def train_model(self, s_batch, target_batch, y_batch, adv_batch):
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        target_batch = torch.FloatTensor(target_batch).to(self.device)
        y_batch = torch.LongTensor(y_batch).to(self.device)
        adv_batch = torch.FloatTensor(adv_batch).to(self.device)

        sample_range = np.arange(len(s_batch))

        with torch.no_grad():
            # for multiply advantage
            policy_old, value_old, _ = self.model(s_batch)
            m_old = Categorical(F.softmax(policy_old, dim=-1))
            log_prob_old = m_old.log_prob(y_batch)

        for i in range(epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / batch_size)):
                sample_idx = sample_range[batch_size * j:batch_size * (j + 1)]
                policy, value, _ = self.model(s_batch[sample_idx])
                m = Categorical(F.softmax(policy, dim=-1))
                log_prob = m.log_prob(y_batch[sample_idx])

                ratio = torch.exp(log_prob - log_prob_old[sample_idx])

                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - ppo_eps,
                    1.0 + ppo_eps) * adv_batch[sample_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(
                    value.sum(1), target_batch[sample_idx])

                self.optimizer.zero_grad()
                loss = actor_loss + critic_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), clip_grad_norm)
                self.optimizer.step()


def make_train_data(reward, done, value, next_value):
    discounted_return = np.empty([num_step])

    # Discounted Return
    if use_gae:
        gae = 0
        for t in range(num_step - 1, -1, -1):
            delta = reward[t] + gamma * \
                next_value[t] * (1 - done[t]) - value[t]
            gae = delta + gamma * lam * (1 - done[t]) * gae

            discounted_return[t] = gae + value[t]

        # For Actor
        adv = discounted_return - value

    else:
        running_add = next_value[-1]
        for t in range(num_step - 1, -1, -1):
            running_add = reward[t] + gamma * running_add * (1 - done[t])
            discounted_return[t] = running_add

        # For Actor
        adv = discounted_return - value

    if use_standardization:
        adv = (adv - adv.mean()) / (adv.std() + stable_eps)

    return discounted_return, adv


if __name__ == '__main__':
    env_id = 'BreakoutNoFrameskip-v4'
    tsne = TSNE(n_components=2)

    env = gym.make(env_id)
    input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n  # 2

    if 'Breakout' in env_id:
        output_size -= 1

    env.close()

    writer = SummaryWriter()
    use_cuda = True
    use_gae = True
    is_load_model = False
    is_render = False
    use_standardization = False
    lr_schedule = False
    life_done = True
    use_noisy_net = True

    model_path = 'models/{}.model'.format(env_id)

    lam = 0.95
    num_worker = 16

    num_step = 128
    ppo_eps = 0.1
    epoch = 3
    batch_size = 32
    max_step = 1.15e8

    learning_rate = 0.0003

    stable_eps = 1e-30
    epslion = 0.1
    entropy_coef = 0.01
    alpha = 0.99
    gamma = 0.99
    clip_grad_norm = 0.5

    agent = ActorAgent(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net)

    if is_load_model:
        agent.model.load_state_dict(torch.load(model_path))

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = AtariEnvironment(env_id, is_render, idx, child_conn)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, 4, 84, 84])

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    global_step = 0
    recent_prob = deque(maxlen=10)
    # total_features = []

    s_states = []
    s_features = []

    t_states = []
    t_features = []
    while True:
        total_features = []

        total_state, total_reward, total_done, total_next_state, total_action = [], [], [], [], []
        global_step += (num_worker * num_step)

        for _ in range(num_step):

            actions,f = agent.get_action(states)
            features_ = []
            for parent_conn, action, f in zip(parent_conns, actions, f):
                parent_conn.send(action)
                f = f.data.cpu().numpy()
                features_.append(f)
            features_ = np.stack(features_)
            total_features.append(features_)
            next_states, rewards, dones, real_dones = [], [], [], []
            for parent_conn in parent_conns:
                s, r, d, rd = parent_conn.recv()
                next_states.append(s)
                rewards.append(r)
                dones.append(d)
                real_dones.append(rd)

            # f = f.data.cpu().numpy()
            # features_.append(f)
            next_states = np.stack(next_states)
            rewards = np.hstack(rewards)
            dones = np.hstack(dones)
            real_dones = np.hstack(real_dones)

            total_state.append(states)
            total_next_state.append(next_states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)

            s_states.append(states[sample_env_idx])
            s_features.append(features_[sample_env_idx])

            states = next_states[:, :, :, :]

            sample_rall += rewards[sample_env_idx]
            sample_step += 1

            if real_dones[sample_env_idx]:
                # print(sample_step)
                # print(np.array(s_states).shape)
                # print(np.array(s_features).shape)

                # print(np.array(total_action).shape)

                # features = np.array(features_)
                # print("-----before--------",features.shape)

                # features = features[:,0,:].reshape(features.shape[0], features.shape[2])
                # print("--------------",features.shape)

                feature_points = tsne.fit_transform(s_features)
                feature_points= feature_points.T.reshape(feature_points.shape[0]*feature_points.shape[1])
                t_features.append(feature_points)
                
                with open("feature_data.csv", 'w', newline='') as resultFile:
                    wr = csv.writer(resultFile)
                    wr.writerows(t_features)
                sample_episode += 1
                writer.add_scalar('data/reward', sample_rall, sample_episode)
                writer.add_scalar('data/step', sample_step, sample_episode)
                sample_rall = 0
                sample_step = 0
                s_states = []
                s_features = []

        total_state = np.stack(total_state).transpose(
            [1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
        total_next_state = np.stack(total_next_state).transpose(
            [1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
        total_reward = np.stack(total_reward).transpose().reshape([-1])
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose().reshape([-1])

        value, next_value, policy = agent.forward_transition(
            total_state, total_next_state)

        policy = policy.detach()
        m = F.softmax(policy, dim=-1)
        recent_prob.append(m.max(1)[0].mean().cpu().numpy())
        writer.add_scalar(
            'data/max_prob',
            np.mean(recent_prob),
            sample_episode)

        total_target = []
        total_adv = []
        for idx in range(num_worker):
            target, adv = make_train_data(total_reward[idx * num_step:(idx + 1) * num_step],
                                          total_done[idx *
                                                     num_step:(idx + 1) * num_step],
                                          value[idx *
                                                num_step:(idx + 1) * num_step],
                                          next_value[idx * num_step:(idx + 1) * num_step])
                       
            total_target.append(target)
            total_adv.append(adv)

        
        agent.train_model(
            total_state,
            np.hstack(total_target),
            total_action,
            np.hstack(total_adv))

        # adjust learning rate
        if lr_schedule:
            new_learing_rate = learning_rate - \
                (global_step / max_step) * learning_rate
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = new_learing_rate
                writer.add_scalar('data/lr', new_learing_rate, sample_episode)

        if global_step % (num_worker * num_step * 100) == 0:
            torch.save(agent.model.state_dict(), model_path)
