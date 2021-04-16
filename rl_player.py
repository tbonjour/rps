import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from dqn import ReplayMemory, DQN, EpsilonGreedyStrategy, extract_tensors, QValues, Experience, FixedStrategy
from player import Player
import random
import matplotlib.pyplot as plt
from datetime import datetime
import os

q_max_list = []
checkpoint_folder = './checkpoints/'
if not os.path.exists(checkpoint_folder):
    os.makedirs(checkpoint_folder)
checkpoint_prefix = 'checkpoint_'
checkpoint_suffix = '_' + datetime.now().strftime('%m%d%H%M') + '.tar'


def get_policy(hist):
    return [hist.count(0) / len(hist),
            hist.count(1) / len(hist),
            hist.count(2) / len(hist)]


class RLAgent(Player):
    def __init__(self, name, others=None, last_n=10, load_path=None, checkpoint=5000, fixed_strategy=False,
                 eps_decay=0.00005):
        if others is None:
            others = [1, 2]
        self.others = others
        self.last_n = last_n
        self.prev_points = 0
        self.batch_size = 32
        self.gamma = 0.9
        self.eps_start = 1
        self.eps_end = 0.01
        self.eps_decay = eps_decay
        self.target_update = 100
        self.plot_at = 1000
        self.q_max = []
        self.q_list = []
        self.checkpoint = checkpoint
        self.memory_size = 1000
        self.lr = 0.00001
        self.train = True

        self.input_dim = len(others) * 6
        self.output_dim = 3
        self.current_step = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayMemory(self.memory_size)

        # Initialize the policy and target networks
        self.policy_net = DQN(self.input_dim, self.output_dim).to(self.device)
        self.target_net = DQN(self.input_dim, self.output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        if load_path is not None:
            checkpoint = torch.load(load_path)
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            self.policy_net.eval()
            self.eps_start = 0
            self.eps_end = 0
            self.train = False
        if fixed_strategy:
            self.strategy = FixedStrategy()
        self.strategy = EpsilonGreedyStrategy(self.eps_start, self.eps_end, self.eps_decay)

        # Set the optimizer
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.lr)
        self.loss = None

        # Push to replay memory
        self.prev_state = None
        self.action = None
        self.reward = None
        self.current_state = None

        super().__init__(name)

    def select_action(self, valid_actions, history):
        # print(self.memory.can_provide_sample(self.batch_size))
        if self.memory.can_provide_sample(self.batch_size) and self.train:
            self.train_model()

        if len(history) > self.last_n + 1:
            self.prev_state, self.current_state = self.get_states(history)
            self.reward = self.get_reward()
            if self.action is not None and self.train:
                self.memory.push(Experience(self.prev_state, self.action,
                                            self.current_state, self.reward))
            self.action = self.get_action(valid_actions)
            return self.action.item()
        else:
            return np.random.choice(valid_actions)

    def get_states(self, history):
        prev_state, current_state = [], []
        if len(history) > self.last_n + 1:
            for other in self.others:
                other_history = [i[other] for i in history]
                other_last_n = other_history[-self.last_n:]
                other_last_n_p = other_history[-self.last_n-1:-1]
                other_policy_total = get_policy(other_history)
                other_policy_last_n = get_policy(other_last_n)
                other_policy_total_p = get_policy(other_history[:-1])
                other_policy_last_n_p = get_policy(other_last_n_p)
                prev_state.extend(other_policy_total_p + other_policy_last_n_p)
                current_state.extend(other_policy_total + other_policy_last_n)
        return torch.as_tensor(prev_state).unsqueeze(-2), torch.as_tensor(current_state).unsqueeze(-2)

    def get_reward(self):
        reward = self.points - self.prev_points
        self.prev_points = self.points
        return torch.tensor([reward])

    def get_action(self, valid_actions):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        if rate > random.random():
            # For random, we can pass the allowable_moves vector and choose from it randomly
            action = np.random.choice(valid_actions)
            return torch.tensor([action]).to(self.device)  # explore
        else:
            with torch.no_grad():
                self.q_max.append(self.policy_net(self.current_state).max().item())
                return self.policy_net(self.current_state).max(1)[1].to(self.device)  # exploit

    def train_model(self):
        experiences = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states = extract_tensors(experiences)
        if self.current_step % self.target_update == 0:
            print('UPDATE TARGET NET', self.current_step)
            self.q_list.extend(self.q_max)
            print('Q Max', sum(self.q_max)/self.target_update)
            q_max_list.append(sum(self.q_max)/self.target_update)
            self.q_max = []
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.current_step % self.plot_at == 0:
            e_ = self.memory.memory[-100:]
            batch = Experience(*zip(*e_))
            print('\n', '*'*42)
            print('EXPLORATION RATE', self.strategy.get_exploration_rate(self.current_step))
            print('REWARD', sum(batch.reward).item())
            print('POLICY', get_policy([i.item() for i in batch.action]))
            print('*' * 42, '\n')
            plt.plot(range(len(q_max_list)), q_max_list)
            plt.show()
        if self.current_step % self.checkpoint == 0:
            print('SAVE CHECKPOINT AT', self.current_step)
            checkpoint_path = checkpoint_folder + checkpoint_prefix + str(self.current_step) + checkpoint_suffix
            torch.save({'model_state_dict': self.policy_net.state_dict()}, checkpoint_path)
        current_q_values = QValues.get_current(self.policy_net, states, actions)
        next_q_values = QValues.get_next(self.policy_net, self.target_net, next_states)
        target_q_values = (next_q_values * self.gamma) + rewards
        self.loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
