import time
import torch
import numpy as np
from SerialThread import SerialThread
from Stoc_o1 import DQNAgent, DroneEnv


# 创建环境
env = DroneEnv(render_mode=True)  # 测试时渲染
state_size = 2
action_size = env.action_space.n

# 创建Agent并加载已训练好的模型权重
agent = DQNAgent(state_size=state_size, action_size=action_size)
agent.load_model("dqn_model_episode_500.pth")  # 模型路径

# 创建串口控制对象
st = SerialThread("COM10")  # 替换为正确的串口号
time.sleep(2)  # 等待串口线程稳定

# 起飞
st.send().takeoff(50)
time.sleep(3)

state = env.reset()
done = False
total_reward = 0
step_count = 0
env.render()

while not done:
    # 使用模型进行决策
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = agent.model(state_tensor)
    action = torch.argmax(q_values).item()

    # 将动作映射到无人机的控制指令（此处仅为示例，需要实际指令）
    # 在CommandConstructor中实现了对应方向移动的指令(如left, right等)
    # 若没有，可考虑使用forward/backward,rotate等替代动作。
    if action == 2:
        # Left动作
        st.send().left(50)
    elif action == 3:

        st.send().back(50)
    elif action == 0:
        # Right动作
        st.send().right(50)
    elif action == 1:
        # Up动作
        st.send().forward(45)

      # 给无人机一定时间执行动作

    next_state, reward, done, _ = env.step(action)
    state = next_state
    total_reward += reward
    step_count += 1
    env.render()
    time.sleep(3)


# 测试结束后降落
st.send().land()
time.sleep(2)
st.shutdown()

env.close()

print(f"Test completed. Total reward: {total_reward} in {step_count} steps.")
