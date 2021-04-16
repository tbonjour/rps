from collections import namedtuple

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
from matplotlib import pyplot as plt


def plot_q(q_list, group_size=100, save=True):
    x = [ele for ele in range(int(len(q_list) / group_size)) for i in range(group_size)]
    sns.lineplot(data={'y': q_list[:int(len(q_list) / group_size)*group_size], 'x': x}, x='x', y='y')
    plt.show()


class DQN(nn.Module):
    """ DQN Class"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_dim, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=24)
        self.out = nn.Linear(in_features=24, out_features=out_dim)

    def forward(self, t):
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


# EPSILON GREEDY
class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)


class FixedStrategy:
    def __init__(self, rate=0.1):
        self.rate = rate

    def get_exploration_rate(self, current_step):
        return self.rate


# EXPERIENCE
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)


# EXTRACT TENSORS
def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return t1, t2, t3, t4


# Get current and next q-values #DDQN
class QValues:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    @staticmethod
    def get_next(policy_net, target_net, next_states):
        next_q_values = policy_net(next_states)
        next_q_target_values = target_net(next_states)
        # DDQN
        return next_q_target_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
