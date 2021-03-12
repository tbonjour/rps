from player import Follower


class RPS:
    """
    Simulator for n-player Rock, Paper, Scissors with multiple rounds.
    Representation of actions using integers: Rock - 0, Paper - 1, Scissors - 2
    Rules:
    1. Paper(1) beats Rock(0), Scissors(2) beats Paper(1), Rock(0) beats Scissors
    2. For n-player game the points are decided accordingly:
        a. If all actions are the same, no one gets any points.
        b. If all three distinct actions(rock, paper and scissors) exist, everyone gets one point(for now).
        c. If there are 2 distinct actions in the set of actions selected, the winning action is decided according
           to Rule 1. All winners get one point each.
    3. At the end of the game, the player with the highest points win. (There can be multiple winners in case of a tie)
    """
    def __init__(self, num_rounds, players):
        self.num_rounds = num_rounds
        self.current_round = 0
        self.players = players
        self.valid_actions = [0, 1, 2]  # Integer representation for R P S respectively
        self.history = [[]]

    def play_game(self, verbose=False, collusion=False):
        while self.current_round != self.num_rounds:
            actions = []
            winners = []
            winning_action = 0
            for idx, p in enumerate(self.players):
                if collusion:
                    if not isinstance(p, Follower):
                        action = p.select_action(self.valid_actions, self.history)
                        if p.name == 'Master':  # Hard-coded for now.
                            master_action = action
                    else:
                        action = p.select_action(self.valid_actions, self.history, master_action)
                else:
                    action = p.select_action(self.valid_actions, self.history)
                actions.append(action)
            self.history.append(actions)
            # Select winners based on the actions. For now, each winner just gets a single point.
            action_set = set(actions)
            if len(action_set) == 2:
                if action_set == {0, 1}:
                    winning_action = 1
                elif action_set == {1, 2}:
                    winning_action = 2
                winners = [i for i in range(len(actions)) if actions[i] == winning_action]
            elif len(action_set) == 3:
                winners = range(len(actions))
            for winner in winners:
                self.players[winner].points += 1
            if verbose:
                print('Round', self.current_round)
                print('Actions', actions)
                print('Winners', winners)
                # print('1 beats 0, 2 beats 1, 0 beats 2')
                for player in self.players:
                    print(player)
            self.current_round += 1
        print()
        for player in self.players:
            print(player)
