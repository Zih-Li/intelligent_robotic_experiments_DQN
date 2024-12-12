import gym
from gym import spaces
import numpy as np
import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# 定义深度Q网络
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 增加神经元数量
        self.fc2 = nn.Linear(128, 128)         # 加入多一层
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        self.grid_size = 4
        self.agent_position = [0, 0]
        self.action_space = spaces.Discrete(4)  # 4个动作：左、下、右、上
        self.observation_space = spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=int)

        pygame.init()
        self.window_size = 500
        self.cell_size = self.window_size // self.grid_size
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()

        self.holes = [(0, 3), (1, 1), (3, 1)]
        self.goal_position = (self.grid_size - 1, self.grid_size - 1)
        self.max_steps = 100
        self.visited_positions = {}

    def reset(self):
        self.agent_position = [0, 0]
        self.current_steps = 0
        self.visited_positions = {}
        self.visited_positions[tuple(self.agent_position)] = 1
        return np.array(self.agent_position)

    def step(self, action):
        x, y = self.agent_position
        prev_position = (x, y)  # 记录上一步位置

        if action == 0:  # Left
            x = max(x - 1, 0)
        elif action == 1:  # Down
            y = min(y + 1, self.grid_size - 1)
        elif action == 2:  # Right
            x = min(x + 1, self.grid_size - 1)
        elif action == 3:  # Up
            y = max(y - 1, 0)

        self.agent_position = [x, y]
        reward = 0

        if tuple(self.agent_position) in self.holes:
            reward = -1
            done = True
        elif tuple(self.agent_position) == self.goal_position:
            reward = 20
            done = True
        else:
            done = False

        # 如果撞墙
        if prev_position == (x, y):
            reward -= 1

        # 重复访问扣分
        if tuple(self.agent_position) in self.visited_positions:
            reward -= 0.1
        else:
            self.visited_positions[tuple(self.agent_position)] = 1

        # 步数限制
        self.current_steps += 1
        if self.current_steps >= self.max_steps:
            done = True
            reward -= 1

        return np.array(self.agent_position), reward, done, {}

    def render(self):
        pygame.event.pump()
        self.window.fill((135, 206, 235))

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                if (x, y) in self.holes:
                    color = (0, 0, 0)
                elif (x, y) == self.goal_position:
                    color = (34, 139, 34)
                else:
                    color = (176, 224, 230)
                pygame.draw.rect(self.window, color, rect)
                pygame.draw.rect(self.window, (255, 255, 255), rect, 1)

        agent_x, agent_y = self.agent_position
        pygame.draw.circle(self.window, (240, 128, 128),
                           (agent_x * self.cell_size + self.cell_size // 2,
                            agent_y * self.cell_size + self.cell_size // 2),
                           self.cell_size // 3)

        pygame.display.flip()
        self.clock.tick(5)

    def close(self):
        pygame.quit()

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)  # 目标网络
        self.target_model.load_state_dict(self.model.state_dict())  # 初始化目标网络
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target_every = 10  # 目标网络更新频率
        self.update_counter = 0  # 计数器

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)  # 转换为张量
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target += self.gamma * torch.max(self.target_model(next_state)).item()
            target_f = self.model(torch.FloatTensor(state).unsqueeze(0))
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(torch.FloatTensor(state).unsqueeze(0)), target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 更新目标网络
        self.update_counter += 1
        if self.update_counter % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

# 初始化环境和代理
env = DroneEnv()
agent = DQNAgent(state_size=2, action_size=env.action_space.n)

num_episodes = 500
batch_size = 32
reward_list = []  # 记录奖励

# 训练循环
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    step_count = 0
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        step_count += 1

        if step_count % 100 == 0:
            env.render()

    agent.replay(batch_size)
    reward_list.append(total_reward)  # 记录回合总奖励
    print(f"Episode {episode + 1}: Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

    # 每隔100个回合保存一次模型
    if (episode + 1) % 100 == 0:
        agent.save_model(f"dqn_model_episode_{episode + 1}.pth")

# 测试代理
num_test_episodes = 100
print("Testing...")  # 开始测试时输出提示信息
for episode in range(num_test_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    step_count = 0
    while not done:
        action = agent.act(state)  # 使用已训练的代理选择动作
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward

        if step_count % 100 == 0:
            env.render()

    # 输出测试结果
    if total_reward < 0:  # 如果总奖励为负数，说明掉进了冰窟
        print(f"Test Episode {episode + 1}: 失败")
    else:  # 如果成功到达终点
        print(f"Test Episode {episode + 1}: 成功, 总奖励: {total_reward}")

env.close()
