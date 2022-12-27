"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/
Dependencies:
torch: 0.4
gym: 0.8.1
numpy

# 实际运行
torch               1.4.0+cpu
torchvision         0.5.0+cpu
gym                 0.26.2
gym-notices         0.0.8
pygame              2.1.0
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import maze_env

# 训练目标：连续 TARGET_TIMES 次找到宝藏
TARGET_TIMES = 10
EDUCATE_FIRST = True # 是否先用正确的步骤训练一下

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.6               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 200   # target update frequency
MEMORY_CAPACITY = 200

env = maze_env.Maze(is_quick=True)
N_ACTIONS = len(env.action_space)
N_STATES = 4 # 神经网络输入状态坐标维度


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()


def update():
    list_textbook = []
    if EDUCATE_FIRST:
        df_textbook = pd.read_csv("../05_QL_maze_env2/success_actions.csv", index_col=0)
        for row in df_textbook.itertuples():
            list_textbook.append(row[2].split('-')) # actions: 3,2,0,1,1,1,2,2,0,0,2,0,2,2,1,1,1,1

    print('\nCollecting experience...')
    episode_rewards = [] # 记录每步的reward
    success_num = 0
    for episode in range(1400):
        s = env.reset()
        ep_r = 0
        i_step = 0
        while True:
            env.render()
            action = dqn.choose_action(s)
            if EDUCATE_FIRST and len(list_textbook) > episode:
                action = int(list_textbook[episode][i_step])

            # take action
            s_, reward, done = env.step(action)

            dqn.store_transition(s, action, reward, s_)

            ep_r += reward
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    pass
                    # print('Ep: ', episode,
                    #     '| Ep_r: ', round(ep_r, 2))
            if done:
                if reward == 1:
                    print([f'[{episode}] reward={reward}'])
                    success_num += 1
                if episode > len(list_textbook):
                    episode_rewards.append(reward)
                break
            s = s_
            i_step += 1

        # 检查连续TARGET_TIMES次都找到宝藏的回合数
        if episode > len(list_textbook) and len(episode_rewards) > TARGET_TIMES and episode_rewards[-TARGET_TIMES:] == [1] * TARGET_TIMES:
            print(f'连续 {TARGET_TIMES} 次找到宝藏，共训练 {episode - len(list_textbook)} 次，踩坑 {episode_rewards.count(-1)} 次')
            break
    # end of game
    print('game over.', f'success {success_num} times.')
    env.destroy()

if __name__ == "__main__":

    env.after(100, update)
    env.mainloop()
