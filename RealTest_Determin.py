import time
import torch
import numpy as np
from SerialThread import SerialThread  # 假设SerialThread相关类已在SerialThread.py中
from Determin import DroneEnv, WinningAgent



# 创建串口控制对象
st = SerialThread("COM10")  # 请替换为正确的串口号
time.sleep(2)  # 等待串口线程稳定

# 起飞
st.send().takeoff(70)
time.sleep(3)

winning_sequence = [2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1]

# Initialize environment and agent
env = DroneEnv()
agent = WinningAgent(winning_sequence)

s = env.reset()
env.render()
time.sleep(3)
done = False
distance = 50
while not done:
    a = agent.choose_action()  # 选择胜利序列中的动作
    if a is not None:  # 确保在序列内
        s, r, done, _ = env.step(a)
        env.render()
        if a == 2:
            # Left动作
            st.send().left(50)
        elif a == 3:

            st.send().back(50)
        elif a == 0:
            # Right动作
            st.send().right(50)
        elif a == 1:
            # Up动作
            st.send().forward(45)
        if done:
            if r > 0:
                print("成功到达终点！")
            else:
                print("进入冰窟，任务失败！")
        time.sleep(3)  # 给无人机一定时间执行动作
    else:
        break  # 胜利序列已完结，退出循环

st.send().land()
time.sleep(2)
st.shutdown()
env.close()
