"""
Sarsa is a online updating method for Reinforcement learning.

Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.

You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""
import pandas as pd
from maze_env import Maze
from RL_brain import SarsaLambdaTable

# 训练目标：连续 TARGET_TIMES 次找到宝藏
TARGET_TIMES = 10
EDUCATE_FIRST = True # 是否先用正确的步骤训练一下

def update():
    list_textbook = []
    if EDUCATE_FIRST:
        df_textbook = pd.read_csv("../05_QL_maze_env2/success_actions.csv", index_col=0)
        for row in df_textbook.itertuples():
            list_textbook.append(row[2].split('-')) # actions: 3,2,0,1,1,1,2,2,0,0,2,0,2,2,1,1,1,1

    episode_rewards = [] # 记录每步的reward
    for episode in range(1000):
        # initial observation
        observation = env.reset()

        # RL choose action based on observation
        action = RL.choose_action(str(observation))
        if EDUCATE_FIRST and len(list_textbook) > episode:
            action = int(list_textbook[episode][0])

        # initial all zero eligibility trace
        RL.eligibility_trace *= 0
        
        i_step = 1
        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))
            if EDUCATE_FIRST and len(list_textbook) > episode:
                try:
                    action_ = int(list_textbook[episode][i_step])
                except:
                    action_ = int(list_textbook[episode][i_step-1])

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation and action
            observation = observation_
            action = action_
            i_step += 1

            # break while loop when end of this episode
            if done:
                if reward == 1:
                    print([f'[{episode}] reward={reward}'])
                    # success_num += 1
                if episode > len(list_textbook):
                    episode_rewards.append(reward)
                break

        # 检查连续TARGET_TIMES次都找到宝藏的回合数
        if episode > len(list_textbook) and len(episode_rewards) > TARGET_TIMES and episode_rewards[-TARGET_TIMES:] == [1] * TARGET_TIMES:
            print(f'连续 {TARGET_TIMES} 次找到宝藏，共训练 {episode - len(list_textbook)} 次，踩坑 {episode_rewards.count(-1)} 次')
            break
    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze(is_quick=True)
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()