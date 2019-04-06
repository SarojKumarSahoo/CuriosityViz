import gym
import os
import random
import numpy as np
import csv
import cv2
from collections import deque

class env():
    def __init__(self, args, h_size = 4, h = 84, w = 84):
        super(env, self).__init__()
        self.env_id = args.env_id
        self.env = gym.make(self.env_id)
        self.is_render = args.is_render
        self.steps = 0
        self.episode = 0
        self.rewards_all = 0
        
        self.h_size = h_size
        self.history = np.zeros([h_size, h, w])
        self.atari_history = np.zeros([h_size, 210, 160, 3])
        self.h = h
        self.w = w 
        self.actions_list = []
        self.actions_map = ['noop', 'fire', 'right', 'left']
        self.total_actions_list = []
        self.rewards_list = []
        self.total_rewards_list = []
        self.history_size = self.h_size
        self.history = np.zeros([self.h_size, h, w])
        self.episode_data = []
        self.r_all_list = []

        self.reset()
        self.lives = self.env.env.ale.lives()
        self.recent_rlist = deque(maxlen=100)

    def run(self, agent, states):
        while True:
            if self.is_render:
                self.env.render()

            actions = agent.get_action(states)
            nactions = actions.copy()
            actions += 1
            _, reward, done, info = self.env.step(int(actions))
            self.actions_list.append(self.actions_map[actions])
            self.rewards_list.append(reward)

            self.history[:3, :, :] = self.history[1:, :, :]
            self.atari_history[:3, :, :, :] = self.atari_history[1:, :, :, :]

            self.history[3, :, :] = self.preprocess(self.env.env.ale.getScreenGrayscale().squeeze().astype('float32'))
            
            self.atari_history[3, :, :, :] = self.env.env.ale.getScreenRGB()

            self.rewards_all += reward
            self.steps += 1

            if done:
                self.recent_rlist.append(self.rewards_all)
                print("[Episode {}] Step: {}  Reward: {} Recent Reward: {} ".format(
                    self.episode, self.steps, self.rewards_all, np.mean(self.recent_rlist)))

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
                self.history, self.atari_history = self.reset()

            return [self.history[:,:,:], reward, done, nactions, self.atari_history[:,:,:,:]]

    
    def reset(self):
        self.steps = 0
        self.episode += 1
        self.env.reset()
        self.init_history(self.env.env.ale.getScreenGrayscale().squeeze().astype('float32'))
        self.init_atari_history(self.env.env.ale.getScreenRGB())

        self.rewards_all = 0
        self.lives = self.env.env.ale.lives()

        return self.history, self.atari_history


    def preprocess(self, img):
        img = cv2.resize(img, (self.h, self.w))
        img *= (1.0/ 255.0)

        return img

    def init_history(self, s):
        for i in range(self.h_size):
            self.history[i, :, :] = self.preprocess(s)

    def init_atari_history(self, s):
        for i in range(self.h_size):
            self.atari_history[i, :, :, :] = s
            