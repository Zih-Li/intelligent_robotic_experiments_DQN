import gym
from gym import spaces
import numpy as np
import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# 定义一个深度Q网络 (DQN) 模型，用于近似Q函数
# DQN是一个简单的三层全连接神经网络：
# 输入层维度为input_dim（状态维度），输出层维度为output_dim（动作空间维度）
# 中间两层为隐藏层，每层包含128个神经元，并使用ReLU激活函数
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)   # 第一隐藏层
        self.fc2 = nn.Linear(128, 128)         # 第二隐藏层
        self.fc3 = nn.Linear(128, output_dim)  # 输出层，对每个动作输出Q值

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 对输入进行第一层线性变换+ReLU
        x = torch.relu(self.fc2(x))  # 第二层线性变换+ReLU
        x = self.fc3(x)              # 输出层线性变换，不使用激活函数
        return x


# 自定义环境类，继承自gym.Env，用于构建一个4x4的网格环境，其中智能体需要从起点(0,0)移动到终点(grid_size-1, grid_size-1)
# 同时存在一些陷阱点(holes)，如果智能体掉入陷阱则失败。
class DroneEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode=True):
        super(DroneEnv, self).__init__()
        self.grid_size = 4
        self.agent_position = [0, 0]  # 初始位置在左上角 (0,0)

        # 动作空间：4个动作，分别对应 左(0), 下(1), 右(2), 上(3)
        self.action_space = spaces.Discrete(4)

        # 状态空间：智能体位置的(x,y)，范围在[0, grid_size-1]
        # 使用Box类型，低值为0，高值为grid_size-1，shape为(2,),类型为float32
        self.observation_space = spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.float32)

        self.render_mode = render_mode
        # 如果需要渲染，则初始化pygame窗口
        if self.render_mode:
            pygame.init()
            self.window_size = 500
            self.cell_size = self.window_size // self.grid_size
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
        else:
            self.window = None
            self.clock = None

        # 定义陷阱和目标位置
        self.holes = [(0, 3), (1, 1), (3, 1)]
        self.goal_position = (self.grid_size - 1, self.grid_size - 1)

        # 最大步数限制，防止无限循环
        self.max_steps = 100
        # 记录访问过的位置，以惩罚重复访问（防止陷入局部循环）
        self.visited_positions = {}

    def reset(self):
        # 重置环境状态
        self.agent_position = [0, 0]
        self.current_steps = 0
        self.visited_positions = {}
        self.visited_positions[tuple(self.agent_position)] = 1
        # 返回初始状态
        return np.array(self.agent_position, dtype=np.float32)

    def step(self, action):
        # 根据动作对智能体位置进行更新
        x, y = self.agent_position
        prev_position = (x, y)  # 记录执行动作前的位置，用于判断是否撞墙

        # 根据动作更新位置（确保不越界）
        if action == 0:  # Left
            x = max(x - 1, 0)
        elif action == 1:  # Down
            y = min(y + 1, self.grid_size - 1)
        elif action == 2:  # Right
            x = min(x + 1, self.grid_size - 1)
        elif action == 3:  # Up
            y = max(y - 1, 0)

        self.agent_position = [x, y]
        reward = 0 # 初始化本步得分

        # 根据新的位置判断奖励或惩罚：
        # 如果掉入陷阱（holes），给予负分并结束回合
        if tuple(self.agent_position) in self.holes:
            reward = -1
            done = True
        # 如果到达目标点(右下角)，给予高额奖励并结束回合
        elif tuple(self.agent_position) == self.goal_position:
            reward = 20
            done = True
        else:
            done = False

        # 如果位置未发生变化（说明动作无效，比如撞墙），则给予惩罚
        if prev_position == (x, y):
            reward -= 1

        # 如果重复访问同一位置，则额外扣分(-0.1)，防止智能体在局部循环中徘徊
        if tuple(self.agent_position) in self.visited_positions:
            reward -= 0.1
        else:
            self.visited_positions[tuple(self.agent_position)] = 1

        # 如果达到最大步数仍未结束，则强制结束并扣分
        self.current_steps += 1
        if self.current_steps >= self.max_steps:
            done = True
            reward -= 1

        # 返回：(状态, 奖励, 是否终止, 额外信息)
        return np.array(self.agent_position, dtype=np.float32), reward, done, {}

    def render(self):
        # 渲染当前环境状态，只在render_mode为True时生效
        if not self.render_mode:
            return
        pygame.event.pump()  # 处理pygame事件，防止窗口卡死
        self.window.fill((135, 206, 235))  # 使用天蓝色填充背景

        # 绘制网格方块及目标、陷阱等
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                if (x, y) in self.holes:
                    color = (0, 0, 0)  # 陷阱用黑色表示
                elif (x, y) == self.goal_position:
                    color = (34, 139, 34)  # 终点用绿色表示
                else:
                    color = (176, 224, 230)  # 普通格子用浅蓝色表示
                pygame.draw.rect(self.window, color, rect)
                pygame.draw.rect(self.window, (255, 255, 255), rect, 1)  # 绘制格子边框

        # 绘制智能体，使用一个红色圆形表示
        agent_x, agent_y = self.agent_position
        pygame.draw.circle(
            self.window, (240, 128, 128),
            (agent_x * self.cell_size + self.cell_size // 2,
             agent_y * self.cell_size + self.cell_size // 2),
            self.cell_size // 3
        )

        pygame.display.flip()     # 更新屏幕内容
        self.clock.tick(5)        # 控制渲染帧率

    def close(self):
        # 关闭pygame窗口
        if self.render_mode:
            pygame.quit()


# 定义DQN智能体类(DQNAgent)
# 功能：
# 1. 维护训练经验缓冲区(memory)
# 2. 基于DQN结构对状态-动作价值进行估计
# 3. 利用ε-贪心策略进行探索与利用平衡
# 4. 使用目标网络(target_model)稳定训练
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size    # 状态维度
        self.action_size = action_size  # 动作空间大小
        self.memory = deque(maxlen=2000) # 经验回放缓存区，存储最近2000条经验
        self.gamma = 0.95    # 折扣因子γ，用于计算未来奖励
        self.epsilon = 1.0   # 初始探索率ε
        self.epsilon_decay = 0.995 # ε衰减率
        self.epsilon_min = 0.01    # 最小探索率
        self.learning_rate = 0.001 # 学习率
        # 主网络和目标网络
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        # 将目标网络的参数与主网络对齐
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.update_target_every = 10  # 每隔多少次更新目标网络一次
        self.update_counter = 0        # 用于计数当前更新次数

    def remember(self, state, action, reward, next_state, done):
        # 将一条经验（状态、动作、奖励、下个状态、是否结束）加入到缓冲区
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # 利用ε-贪心策略选择动作
        # 以概率ε随机选动作，以(1-ε)的概率选择Q值最大的动作
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()  # 返回Q值最大的动作索引

    def replay(self, batch_size):
        # 从记忆中随机抽取一批数据进行训练
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        # 拆分batch数据
        states = torch.FloatTensor([m[0] for m in minibatch])
        actions = torch.LongTensor([m[1] for m in minibatch]).unsqueeze(-1)
        rewards = torch.FloatTensor([m[2] for m in minibatch])
        next_states = torch.FloatTensor([m[3] for m in minibatch])
        dones = torch.FloatTensor([float(m[4]) for m in minibatch])

        # 当前状态下的Q值分布
        q_values = self.model(states)
        # 根据实际采取的动作actions选择Q值
        q_values = q_values.gather(1, actions)

        # 使用目标网络计算下一个状态的最大Q值
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(dim=1)[0]
            # 计算目标值：r + (1-done)*γ*maxQ(next_state)
            target = rewards + (1 - dones) * self.gamma * next_q_values

        # 调整target维度与q_values匹配
        target = target.unsqueeze(1)

        # 使用MSELoss计算损失并反向传播更新主网络参数
        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 衰减探索率ε
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 定期更新目标网络参数
        self.update_counter += 1
        if self.update_counter % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        # 保存主网络参数到指定文件
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        # 从文件中加载模型参数
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


if __name__ == '__main__':
    # 仅在直接运行本文件时执行下列代码，如果作为模块被import，则不执行

    # 初始化环境（不渲染）和智能体
    env = DroneEnv(render_mode=False)
    agent = DQNAgent(state_size=2, action_size=env.action_space.n)

    num_episodes = 500
    batch_size = 32
    reward_list = []  # 用于记录每个回合的总奖励

    # 训练循环
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        step_count = 0
        while not done:
            # 根据当前策略选择动作
            action = agent.act(state)
            # 执行动作并观察下一个状态和奖励
            next_state, reward, done, _ = env.step(action)
            # 将经验存储到replay buffer中
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1

        # 使用小批量数据进行训练
        agent.replay(batch_size)
        reward_list.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

        # 每隔100回合保存模型
        if (episode + 1 == 500):
            agent.save_model(f"dqn_model_episode_{episode + 1}.pth")

    # 测试训练得到的智能体，渲染环境观察行为
    test_env = DroneEnv(render_mode=True)
    num_test_episodes = 10
    print("Testing...")
    for episode in range(num_test_episodes):
        state = test_env.reset()
        test_env.render()
        done = False
        total_reward = 0
        step_count = 0
        while not done:
            # 测试时使用贪心策略（不探索），直接选择Q值最高的动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = agent.model(state_tensor)
            action = torch.argmax(q_values).item()

            next_state, reward, done, _ = test_env.step(action)
            state = next_state
            total_reward += reward
            step_count += 1

            # 每步渲染，查看智能体移动情况
            test_env.render()

        if total_reward < 0:
            print(f"Test Episode {episode + 1}: 失败, 总奖励: {total_reward}")
        else:
            print(f"Test Episode {episode + 1}: 成功, 总奖励: {total_reward}")

    test_env.close()
