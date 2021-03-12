import numpy as np


class Player:
    def __init__(self, name):
        self.name = name
        self.points = 0

    def __repr__(self):
        return 'Name: ' + str(self.name) + '  Points: ' + str(self.points)

    def select_action(self, valid_actions, history):
        pass


class RandomPlayer(Player):
    def select_action(self, valid_actions, history):
        return np.random.choice(valid_actions)


class FixedPlayer(Player):
    def __init__(self, name, policy=None):
        super().__init__(name)
        self.policy = policy

    def select_action(self, valid_actions, history):
        if self.policy:
            return np.random.choice(valid_actions, p=self.policy)
        else:
            return np.random.choice(valid_actions)


class Follower(Player):
    def __init__(self, name, collusion_prob=0):
        super().__init__(name)
        self.collusion_prob = collusion_prob

    def select_action(self, valid_actions, history, master_move=None):
        collude = np.random.choice([0, 1], p=[1-self.collusion_prob, self.collusion_prob])
        if collude:
            return[2, 0, 1][master_move]
        else:
            return np.random.choice(valid_actions)
