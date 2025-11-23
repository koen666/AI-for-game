# dqn_cartpole.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque

# 超参数
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 50000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE = 50
NUM_EPISODES = 500

# ---------------------------------------------------
# 1. 定义 Q 网络（神经网络近似 Q(s,a)）
# ---------------------------------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------
# 2. 经验回放池
# ---------------------------------------------------
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def push(self, s, a, r, s_, done):
        self.buffer.append((s, a, r, s_, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, done = zip(*batch)
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(s_, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------
# 3. 选择动作（ε-greedy）
# ---------------------------------------------------
def select_action(state, eps, policy_net, action_dim):
    if random.random() < eps:
        return random.randint(0, action_dim - 1)
    else:
        with torch.no_grad():
            q_values = policy_net(torch.tensor(state, dtype=torch.float32))
            return int(torch.argmax(q_values))


# ---------------------------------------------------
# 主程序：DQN 训练 CartPole
# ---------------------------------------------------
env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())  # 初始同步

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_SIZE)

epsilon = EPS_START
rewards_history = []

for episode in range(1, NUM_EPISODES + 1):
    state, _ = env.reset()
    total_reward = 0

    for step in range(500):
        action = select_action(state, epsilon, policy_net, action_dim)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory.push(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        # ----------------------
        # 开始训练（需要足够经验）
        # ----------------------
        if len(memory) > BATCH_SIZE:
            s, a, r, s_, done_batch = memory.sample(BATCH_SIZE)

            # Q(s,a)
            q_values = policy_net(s)
            q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)

            # 目标 Q
            with torch.no_grad():
                next_q = target_net(s_).max(1)[0]
                target = r + GAMMA * next_q * (1 - done_batch)

            loss = nn.MSELoss()(q_value, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    rewards_history.append(total_reward)

    # ε 衰减
    epsilon = max(EPS_END, epsilon * EPS_DECAY)

    # 每隔 TARGET_UPDATE 次同步目标网络
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # 打印进度
    if episode % 20 == 0:
        avg_reward = np.mean(rewards_history[-20:])
        print(f"Episode {episode}, avg reward = {avg_reward:.2f}, epsilon={epsilon:.2f}")

env.close()
print("训练完成！")
