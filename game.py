import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from replay_memory import *
from policy_net import *
from utils import *

from config import *


env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()



# 获取屏幕大小，以便我们可以根据AI gym返回的形状正确初始化图层。 
# 此时的典型尺寸接近3x40x90
# 这是get_screen（）中的限幅和缩小渲染缓冲区的结果
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# 从gym行动空间中获取行动数量
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1)将返回每行的最大列值。 
            # 最大结果的第二列是找到最大元素的索引，因此我们选择具有较大预期奖励的行动。
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 取100个episode的平均值并绘制它们
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # 暂停一下，以便更新图表
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # 转置batch（有关详细说明，请参阅https://stackoverflow.com/a/19343/3343043）。
    # 这会将过渡的batch数组转换为batch数组的过渡。
    batch = Transition(*zip(*transitions))

    # 计算非最终状态的掩码并连接batch元素（最终状态将是模拟结束后的状态）
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 计算Q(s_t，a) - 模型计算Q(s_t)，然后我们选择所采取的动作列。
    # 这些是根据policy_net对每个batch状态采取的操作
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 计算所有下一个状态的V(s_{t+1})
    # non_final_next_states的操作的预期值是基于“较旧的”target_net计算的; 
    # 用max(1)[0]选择最佳奖励。这是基于掩码合并的，这样我们就可以得到预期的状态值，或者在状态是最终的情况下为0。
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # 计算预期的Q值
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # 计算Huber损失
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # 优化模型
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 50
for i_episode in range(num_episodes):
    # 初始化环境和状态
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
        # 选择动作并执行
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # 观察新的状态
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # 在记忆中存储过渡
        memory.push(state, action, next_state, reward)

        # 移动到下一个状态
        state = next_state

        # 执行优化的一个步骤（在目标网络上）
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # 更新目标网络，复制DQN中的所有权重和偏差
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()