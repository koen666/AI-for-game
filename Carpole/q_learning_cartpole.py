# q_learning_cartpole.py
import gymnasium as gym
import numpy as np
import math
from collections import defaultdict
import pickle

# ---------- 超参数 ----------
NUM_EPISODES = 1000
MAX_STEPS = 500          # CartPole-v1 最多步数
ALPHA = 0.1              # 学习率
GAMMA = 0.99             # 折扣因子
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995        # 每个 episode 衰减因子
NUM_BINS = (10, 10, 10, 10)  # 每个状态维度的分箱数量
PRINT_EVERY = 50
# ---------------------------

# 使用 gymnasium
env = gym.make("CartPole-v1")

# 定义每维的分箱边界
obs_space_bounds = list(zip(
    [-4.8, -4.0, -0.418, -4.0],  # position, velocity, angle, angular velocity
    [4.8,  4.0,  0.418,  4.0]
))

# 创建分箱
def create_bins(num_bins, obs_space_bounds):
    bins = []
    for i in range(len(num_bins)):
        low, high = obs_space_bounds[i]
        bins.append(np.linspace(low, high, num_bins[i] - 1))
    return bins

BINS = create_bins(NUM_BINS, obs_space_bounds)

def discretize(obs):
    """将连续变量 obs 转成离散 bins"""
    discretized = []
    for i, val in enumerate(obs):
        discretized.append(int(np.digitize(val, BINS[i])))
    return tuple(discretized)

# Q 表
Q = defaultdict(lambda: np.zeros(env.action_space.n))

def choose_action(state, eps):
    if np.random.rand() < eps:
        return env.action_space.sample()
    else:
        return int(np.argmax(Q[state]))

# 训练
epsilon = EPS_START
rewards_all = []

for episode in range(1, NUM_EPISODES + 1):
    obs, info = env.reset()
    state = discretize(obs)
    total_reward = 0

    for t in range(MAX_STEPS):

        action = choose_action(state, epsilon)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        next_state = discretize(next_obs)

        # Q-learning 更新
        best_next = np.max(Q[next_state])
        Q[state][action] += ALPHA * (reward + GAMMA * best_next - Q[state][action])

        state = next_state
        total_reward += reward

        if done:
            break

    # epsilon 衰减
    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    rewards_all.append(total_reward)

    if episode % PRINT_EVERY == 0:
        avg_last = np.mean(rewards_all[-PRINT_EVERY:])
        print(f"Episode {episode}\tAvg reward (last {PRINT_EVERY}): "
              f"{avg_last:.2f}\tEpsilon: {epsilon:.3f}")

# 保存 Q 表
with open("q_table_cartpole.pkl", "wb") as f:
    pickle.dump(dict(Q), f)

print("Training finished. Example rewards (last 20):", rewards_all[-20:])
env.close()
