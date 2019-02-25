import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class ActorCriticNetwork(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCriticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc = nn.Linear(7 * 7 * 64, 512)
        num_outputs = action_space 

        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_outputs)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a = 1.0)
                p.bias.data.zero_()

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(input.size(0), -1)
        x = F.leaky_relu(self.fc(x))
        

        return self.actor_linear(x), self.critic_linear(x)