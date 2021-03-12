Simulator for n-player Rock, Paper, Scissors with multiple rounds.

Actions are represented using integers: Rock - 0, Paper - 1, Scissors - 2.
    
### Rules:
1. Paper(1) beats Rock(0), Scissors(2) beats Paper(1), Rock(0) beats Scissors
2. For n-player game the points are decided accordingly:
    a. If all actions are the same, no one gets any points.
    b. If all three distinct actions(rock, paper and scissors) exist, everyone gets one point(for now).
    c. If there are 2 distinct actions in the set of actions selected, the winning action is decided according
       to Rule 1. All winners get one point each.
3. At the end of the game, the player with the highest points win. (There can be multiple winners in case of a tie)

### Guide:
- The code for the gameplay of rock, paper, scissors is implemented in rps.py
- To create your own player, extend player.Player and include the select_actions(valid_actions, history) function. Some simple players have been implemented in player.py for reference.
- To run a game, intialize a list of players. Call rps.play_game(num_rounds, players). A few examples of running games is provided in main.py.

