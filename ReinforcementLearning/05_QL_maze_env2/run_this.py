"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import pandas as pd
from maze_env import Maze
from RL_brain import QLearningTable


def update():
    success_actions = []
    max_episode = 100
    for episode in range(max_episode):
        # initial observation
        observation = env.reset()

        action_path = []
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))
            action_path.append(str(action))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                if reward == 1:
                    row = [f'[{episode}/{max_episode}]', '-'.join(action_path)]
                    print(row)
                    success_actions.append(row)
                break


    pd.DataFrame(success_actions, columns=['episode', 'actions']).to_csv('success_actions.csv')
    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze(is_quick=True)
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()

    RL.print_q_table('Q-Table.csv')