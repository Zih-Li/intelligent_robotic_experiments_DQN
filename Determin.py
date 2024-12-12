import gym
from gym import spaces
import numpy as np
import pygame

class DroneEnv(gym.Env):
    def __init__(self):
        self.grid_size = 8
        self.agent_position = [0, 0]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=int)

        # Initialize pygame
        self.window_size = 500  # 更大的窗口尺寸
        self.cell_size = self.window_size // self.grid_size
        pygame.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()

        # Define holes and goal positions
        self.holes = [(3, 3), (2, 2),(6,1), (3,7),(0, 6), (0,2),(4,5)]
        self.goal_position = (self.grid_size - 1, self.grid_size - 1)

    def reset(self):
        self.agent_position = [0, 0]
        return np.array(self.agent_position)

    def step(self, action):
        x, y = self.agent_position
        if action == 0:   # Left
            x = max(x - 1, 0)
        elif action == 1: # Down
            y = min(y + 1, self.grid_size - 1)
        elif action == 2: # Right
            x = min(x + 1, self.grid_size - 1)
        elif action == 3: # Up
            y = max(y - 1, 0)

        self.agent_position = [x, y]

        if tuple(self.agent_position) in self.holes:
            reward = -1
            done = True
        elif tuple(self.agent_position) == self.goal_position:
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        return np.array(self.agent_position), reward, done, {}

    def render(self):
        self.window.fill((135, 206, 235))  # Fill the background with sky blue

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                if (x, y) in self.holes:
                    color = (0, 0, 0)  # Dark color for holes
                elif (x, y) == self.goal_position:
                    color = (34, 139, 34)  # Green for the goal
                else:
                    color = (176, 224, 230)  # Light blue for empty spaces
                pygame.draw.rect(self.window, color, rect)
                pygame.draw.rect(self.window, (255, 255, 255), rect, 1)  # Grid lines

        # Draw the agent as a circle
        agent_x, agent_y = self.agent_position
        pygame.draw.circle(self.window, (240, 128, 128),
                           (agent_x * self.cell_size + self.cell_size // 2,
                            agent_y * self.cell_size + self.cell_size // 2),
                           self.cell_size // 3)

        pygame.display.flip()
        self.clock.tick(5)  # 控制帧率

    def close(self):
        pygame.quit()

class WinningAgent:
    def __init__(self, winning_sequence):
        self.winning_sequence = winning_sequence
        self.current_step = 0

    def choose_action(self):
        if self.current_step < len(self.winning_sequence):
            action = self.winning_sequence[self.current_step]
            self.current_step += 1
            return action
        else:
            return None  # 已经完成胜利序列

# 定义胜利序列（根据环境设定，以达到目标位置）
if __name__ == '__main__':
    winning_sequence = [2, 1, 1, 1, 1, 1, 1, 2, 2,2,2,2,2, 1]

    # Initialize environment and agent
    env = DroneEnv()

    agent = WinningAgent(winning_sequence)

    s = env.reset()
    env.render()
    done = False
    while not done:
        a = agent.choose_action()  # 选择胜利序列中的动作
        if a is not None:  # 确保在序列内
            s, r, done, _ = env.step(a)
            env.render()

            if done:
                if r > 0:
                    print("成功到达终点！")
                else:
                    print("进入冰窟，任务失败！")
        else:
            break  # 胜利序列已完结，退出循环

    env.close()








