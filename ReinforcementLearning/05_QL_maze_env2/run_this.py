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

# 训练目标：连续 TARGET_TIMES 次找到宝藏
TARGET_TIMES = 10

EDUCATE_FIRST = False # 是否先用正确的步骤训练一下
OUTPUT_ACTIONS_CSV = False # 输出成功路径

def update():
    if EDUCATE_FIRST:
        df_textbook = pd.read_csv("success_actions.csv", index_col=0)
        list_textbook = []
        for row in df_textbook.itertuples():
            list_textbook.append(row[2].split('-')) # actions: 3,2,0,1,1,1,2,2,0,0,2,0,2,2,1,1,1,1

    episode_rewards = [] # 记录每步的reward
    success_actions = []
    max_episode = 10000
    for episode in range(max_episode):
        # initial observation
        observation = env.reset()

        action_path = []
        i_step = 0
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))
            if EDUCATE_FIRST and len(list_textbook) > episode:
                action = int(list_textbook[episode][i_step])
            action_path.append(str(action))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_
            i_step += 1

            # break while loop when end of this episode
            if done:
                if reward == 1:
                    row = [f'[{episode}/{max_episode}]', '-'.join(action_path)]
                    print(row)
                    success_actions.append(row)
                
                episode_rewards.append(reward)
                break
        
        # 检查连续TARGET_TIMES次都找到宝藏的回合数
        if len(episode_rewards) > TARGET_TIMES and episode_rewards[-TARGET_TIMES:] == [1] * TARGET_TIMES:
            print(f'连续 {TARGET_TIMES} 次找到宝藏，共训练 {episode} 次，踩坑 {episode_rewards.count(-1)} 次')
            break

    if OUTPUT_ACTIONS_CSV:
        pd.DataFrame(success_actions, columns=['episode', 'actions']).to_csv('success_actions.csv')
    # end of game
    print('game over', f'success {len(success_actions)} times.')
    env.destroy()

if __name__ == "__main__":
    env = Maze(is_quick=True)
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()

    # RL.print_q_table('Q-Table.csv')