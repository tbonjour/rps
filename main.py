from player import RandomPlayer, Follower, FixedPlayer
from rps import RPS


if __name__ == '__main__':
    players = [RandomPlayer('Random A'),
               RandomPlayer('Random B'),
               RandomPlayer('Random C')]
    rps_ = RPS(1000, players)
    rps_.play_game()

    # policy must sum to 1
    players = [FixedPlayer('Fixed A', policy=[0.4, 0.3, 0.3]),
               FixedPlayer('Fixed B', policy=[0.1, 0.3, 0.6]),
               FixedPlayer('Fixed C', policy=[0.3, 0.3, 0.4])]
    rps_ = RPS(1000, players)
    rps_.play_game()

    players = [RandomPlayer('Random'),
               RandomPlayer('Master'),
               Follower('Follower', collusion_prob=0.5)]
    rps_ = RPS(1000, players)
    rps_.play_game(collusion=True)

    players = [FixedPlayer('Fixed A', policy=[0.5, 0.5, 0]),
               RandomPlayer('Master'),
               Follower('Follower', collusion_prob=0.5)]
    rps_ = RPS(1000, players)
    rps_.play_game(collusion=True)
