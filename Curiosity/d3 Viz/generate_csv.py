import csv
import random
import numpy as np

actions = ["left", "right", "no-op", "fire"]

reward = [0, 1, 4, 7]
episodes = 80

episode_actions = []
episode_rewards = []
for j in range(episodes):
    steps = random.randint(300,1000)
    action_list = []
    reward_list = []
    for i in range(steps):
        action_list.append(random.choice(actions))
        reward_list.append(random.choice(reward))
    episode_actions.append(action_list)
    episode_rewards.append(reward_list)

with open("actions.csv", 'w', newline='') as resultFile:
    wr = csv.writer(resultFile)
    wr.writerows(episode_actions)

with open("rewards.csv", 'w', newline='') as resultFile:
    wr = csv.writer(resultFile)
    wr.writerows(episode_rewards)