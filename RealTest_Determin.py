import time
import torch
import numpy as np
from SerialThread import SerialThread  # 假设SerialThread相关类已在SerialThread.py中
from Determin import DroneEnv, WinningAgent



# 创建串口控制对象
st = SerialThread("COM3")  # 请替换为正确的串口号
time.sleep(2)  # 等待串口线程稳定

# 起飞
st.send().takeoff(70)
time.sleep(3)

winning_sequence = [2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1]

# Initialize environment and agent
env = DroneEnv()
agent = WinningAgent(winning_sequence)

s = env.reset()
done = False
distance = 50
while not done:
    a = agent.choose_action()  # 选择胜利序列中的动作
    if a is not None:  # 确保在序列内
        s, r, done, _ = env.step(a)
        env.render()
        if a == 0:
            # Left动作
            # 此处需根据您的CommandConstructor提供的API来编写
            st.send().left(distance)
        elif a == 1:
            # Down动作
            # 下沉或后退等动作需根据实际功能修改
            st.send().back(distance)
        elif a == 2:
            # Right动作
            st.send().right(distance)
        elif a == 3:
            # Up动作
            # 如果没有升高命令，可以考虑takeoff或者前进替换
            # 或者使用pitch/roll控制实现“上移”效果(需自行定义)
            st.send().forward(distance)
        if done:
            if r > 0:
                print("成功到达终点！")
            else:
                print("进入冰窟，任务失败！")
        time.sleep(2)  # 给无人机一定时间执行动作
    else:
        break  # 胜利序列已完结，退出循环

st.send().land()
time.sleep(2)
st.shutdown()
env.close()
