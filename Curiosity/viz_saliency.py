import gym
import os
import random
from itertools import chain

import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import cv2

from model import *
import time

import torch.optim as optim
from torch.multiprocessing import Pipe, Process
import matplotlib.pyplot as plt
from collections import deque
from saliency_helper import *
from scipy.misc import imresize 


def pre_proc(X):
    x = cv2.resize(X, (84, 84))
    x *= (1.0 / 255.0)

    return x


class AtariEnvironment(Process):
    def __init__( self,env_id, is_render,env_idx,child_conn,history_size=4,h=84, w=84):
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

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.atari_history = np.zeros([history_size, 210, 160, 3])
        self.h = h
        self.w = w

        self.reset()
        self.lives = self.env.env.ale.lives()

    def run(self):
        super(AtariEnvironment, self).run()
        while True:
            life_done = False
            action = self.child_conn.recv()
            if self.is_render:
                self.env.render()

            if 'Breakout' in self.env_id:
                action += 1

            _, reward, done, info = self.env.step(action)

            if life_done:
                if self.lives > info['ale.lives'] and info['ale.lives'] > 0:
                    force_done = True
                    self.lives = info['ale.lives']
                else:
                    force_done = done
            else:
                force_done = done

            self.history[:3, :, :] = self.history[1:, :, :]
            self.atari_history[:3, :, :, :] = self.atari_history[1:, :, :, :]

            self.history[3, :, :] = self.pre_proc(
                self.env.env.ale.getScreenGrayscale().squeeze().astype('float32'))
            self.atari_history[3, :, :, :] = self.env.env.ale.getScreenRGB()

            self.rall += reward
            self.steps += 1

            if done:
                self.recent_rlist.append(self.rall)
                print("[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}".format(
                    self.episode, self.env_idx, self.steps, self.rall, np.mean(self.recent_rlist)))

                self.history, self.atari_history = self.reset()

            self.child_conn.send(
                [self.history[:, :, :], reward, force_done, done, self.atari_history[:,:,:,:]])

    def reset(self):
        self.steps = 0
        self.episode += 1
        self.rall = 0
        self.env.reset()
        self.lives = self.env.env.ale.lives()
        self.get_init_state(
            self.env.env.ale.getScreenGrayscale().squeeze().astype('float32'))
        self.init_atari_history(self.env.env.ale.getScreenRGB())

        return self.history[:, :, :], self.atari_history[:,:,:,:]

    def pre_proc(self, X):
        x = cv2.resize(X, (self.h, self.w))
        x *= (1.0 / 255.0)

        return x

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)

    def init_atari_history(self, s):
        for i in range(self.history_size):
            self.atari_history[i, :, :, :] = s


class ActorAgent(object):
    def __init__(self,input_size,output_size,num_env,use_cuda=False):
        self.model = ActorCriticNetwork(
            input_size, output_size)
        self.num_env = num_env
        self.output_size = output_size
        self.input_size = input_size

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00025)

        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.model = self.model.to(self.device)

    def get_action(self, state):
        axis = 1
        state = torch.Tensor(state).to(self.device)
        state = state.float()
        policy, value = self.model(state)
        policy = F.softmax(policy, dim=-1).data.cpu().numpy()
        r = np.expand_dims(np.random.rand(policy.shape[1 - axis]), axis=axis)
        action =  (policy.cumsum(axis=axis) > r).argmax(axis=axis)
        return action

        

if __name__ == '__main__':
    env_id = 'BreakoutDeterministic-v4'
    env = gym.make(env_id)
    input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n  # 2

    if 'Breakout' in env_id:
        output_size -= 1

    env.close()

    use_cuda = True
    is_load_model = True 
    is_render = False
    life_done = True
    use_noisy_net = True

    model_path = 'my-{}.model'.format(env_id)

    num_worker = 1
    num_worker_per_env = 1


  
    agent = ActorAgent(
        input_size,
        output_size,
        num_worker_per_env *
        num_worker,
        use_cuda=use_cuda)

    agent.model.load_state_dict(torch.load(model_path))

    agent.model.eval()

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

    states = np.zeros([num_worker * num_worker_per_env, 4, 84, 84])

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    global_step = 0
    recent_prob = deque(maxlen=10)
    rollout = False
    rd = False
    count = 0
    frames = []
    frames_rgb = []
    if not rollout:
        while not rd:
            total_state, total_reward, total_done, total_next_state, total_action = [], [], [], [], []

            actions = agent.get_action(states)

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            next_states, rewards, dones, real_dones = [], [], [], []
            for parent_conn in parent_conns:
                s, r, d, rd, s_rgb = parent_conn.recv()
                next_states.append(s)
                rewards.append(r)
                dones.append(d)
                real_dones.append(rd)
            
            frames.append(s)
            frames_rgb.append(s_rgb)
            next_states = np.stack(next_states)
            rewards = np.hstack(rewards)
            dones = np.hstack(dones)
            real_dones = np.hstack(real_dones)

            
            count +=1
            states = next_states[:, :, :, :]

    frame = frames[0][3]
    mask = get_mask([20,20], [84,84], r=5)
    p = get_perturbed_image(frames[0][3], mask)
    blurred_image = gaussian_filter(frames[0][3], sigma = 5)

    for i in range(3):
        f = plt.figure(figsize=[6, 6*1.3])
        if i == 0:
            plt.imshow(frame)
            plt.show()
            f.savefig('fframe.jpg')
        elif i == 1:
            plt.imshow(p)
            plt.show()
            f.savefig('p.jpg')
        elif i == 2:
            plt.imshow(blurred_image)
            plt.show()
            f.savefig('b.jpg')


    # actor_score = saliency_score(agent.model, frames[0], 3)
    # frame = saliency_on_frame(actor_score, frame, fudge_factor=200, channel = 2, sigma = 3)


    # for i in range(len(frames)):
    #     if i%100 ==0:
    #         actor_score = saliency_score(agent.model, frames[i], 3)
    #         critic_score = saliency_score(agent.model, frames[i], 1, mode='critic')
            
    #         print(i)
    #         import matplotlib.pyplot as plt
    #         f = plt.figure(figsize=[6, 6*1.3])

    #         frame = frames_rgb[i][3].copy()

    #         frame = saliency_on_frame(actor_score, frame, fudge_factor=200, channel = 2, sigma = 3)
    #         frame = saliency_on_frame(critic_score, frame, fudge_factor=400, channel = 0, sigma = 3)
    #         plt.imshow(frame)
    #         plt.show()
    #         f.savefig('frames/frame{}.png'.format(i))

   