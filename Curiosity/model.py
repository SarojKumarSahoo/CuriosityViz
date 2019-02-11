import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticNetwork(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCriticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm_size = 256
        num_outputs = action_space.n

        self.lstm = nn.LSTMCell(32*5*5, self.lstm_size)

        self.critic_linear = nn.Linear(self.lstm_size, 1)
        self.actor_linear = nn.Linear(self.lstm_size, num_outputs)

        

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        # print (inputs.size())
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, 32*5*5)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)