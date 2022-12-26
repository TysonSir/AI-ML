"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example. The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import os
import numpy as np
import time
import tkinter as tk
from PIL import Image, ImageTk # pip install Pillow

work_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(work_dir)

# DEFAULT_MAZE_MAP = \
# [
#     [3, 0, 0, 0],
#     [0, 0, 1, 0],
#     [0, 0, 0, 0],
#     [0, 0, 0, 2],
# ]
DEFAULT_MAZE_MAP = \
[
    [3, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 2],
]
# DEFAULT_MAZE_MAP = \
# [
#     [3, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
#     [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
#     [0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
#     [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 2],
# ]

UNIT = 40   # pixels
MAZE_H = len(DEFAULT_MAZE_MAP)  # grid height
MAZE_W = len(DEFAULT_MAZE_MAP[0])  # grid width

class Maze(tk.Tk, object):
    def __init__(self, is_quick=False):
        super(Maze, self).__init__()
        self.is_sleep = not is_quick # 控制time.sleep
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('寻宝')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))

        self.hell_list = []
        self.img_list = []
        self._build_maze()

    def _set_image(self, img_path, start_x, start_y, size=UNIT):
        img = Image.open(img_path).resize((size, size), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(img)
        self.img_list.append((img, photo))
        return self.canvas.create_image(start_x, start_y, anchor=tk.NW, image=self.img_list[-1][1])

    def _create_hell(self, origin, x, y):
        # x, y = x - 1, y - 1
        hell_center = origin + np.array([UNIT * y, UNIT * x])
        hell = self.canvas.create_rectangle(
            hell_center[0] - 15, hell_center[1] - 15,
            hell_center[0] + 15, hell_center[1] + 15,
            fill='white', width=0)
        self.hell_list.append(hell)
        self._set_image('./env_img/water.png', hell_center[0] - UNIT/2 + 2, hell_center[1] - UNIT/2 + 2, 36)

    # def _create_human(self, origin, x, y):
    #     human_center = origin + np.array([UNIT * y, UNIT * x])
    #     self.rect = self.canvas.create_rectangle(
    #         human_center[0] - 15, human_center[1] - 15,
    #         human_center[0] + 15, human_center[1] + 15,
    #         width=0
    #     )
    #     self.human = self._set_image('./env_img/human.png', human_center[0] - UNIT/2 + 2, human_center[1] - UNIT/2 + 2, 36)
    
    def _create_human(self, origin):
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            width=0
        )
        self.human = self._set_image('./env_img/human.png', origin[0] - UNIT/2 + 2, origin[1] - UNIT/2 + 2, 36)
    
    def _delete_human(self):
        self.canvas.delete(self.rect)
        self.canvas.delete(self.human)
    
    def _move_human(self, move_x, move_y):
        self.canvas.move(self.rect, move_x, move_y)  # move agent
        self.canvas.move(self.human, move_x, move_y)  # move agent

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1, fill='lightskyblue')
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1, fill='lightskyblue')

        # create origin
        origin = np.array([20, 20])

        # draw hell
        self.human_r_c = (0, 0)
        oval_r_c = (0, 0)
        for row in range(MAZE_H):
            for col in range(MAZE_W):
                if DEFAULT_MAZE_MAP[row][col] == 1:
                    self._create_hell(origin, row, col)
                if DEFAULT_MAZE_MAP[row][col] == 2:
                    oval_r_c = (row, col)
                if DEFAULT_MAZE_MAP[row][col] == 3:
                    self.human_r_c = (row, col)

        # create oval
        oval_x, oval_y = oval_r_c[0], oval_r_c[1]
        oval_center = origin + np.array([UNIT * oval_y, UNIT * oval_x])
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')
        self._set_image('./env_img/goal.png', oval_center[0] - UNIT/2 + 3, oval_center[1] - UNIT/2 + 3, 34)

        # create red rect
        # self._create_human(origin, self.human_r_c[0], self.human_r_c[1])
        self._create_human(origin)

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        if self.is_sleep:
            time.sleep(0.5)

        self._delete_human()

        origin = np.array([20, 20])
        # self._create_human(origin, self.human_r_c[0], self.human_r_c[1])
        self._create_human(origin)

        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self._move_human(base_action[0], base_action[1])

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(hell) for hell in self.hell_list]:
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        if self.is_sleep:
            time.sleep(0.1)
        self.update()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break

if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()