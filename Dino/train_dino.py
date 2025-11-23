# train_dino.py
import time
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from PIL import Image

# ------------------- 超参数 -------------------
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 32
MEMORY_SIZE = 5000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE = 10
NUM_EPISODES = 1000
STATE_SIZE = (80, 80)  # 预处理图像尺寸
# ---------------------------------------------

# ------------------- DQN 网络 -------------------
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(h*w, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, outputs)
        )

    def forward(self, x):
        return self.net(x)

# ------------------- 经验回放 -------------------
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def push(self, s, a, r, s_, done):
        self.buffer.append((s, a, r, s_, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, done = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(np.array(s_), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

# ------------------- 游戏环境 -------------------
class DinoEnv:
    def __init__(self):
        self.driver = webdriver.Chrome()  # 或指定 ChromeDriver 路径
        self.driver.get("chrome://dino")
        time.sleep(2)
        self.body = self.driver.find_element("tag name", "body")
        self.start_game()

    def start_game(self):
        """按空格开始游戏"""
        self.body.send_keys(Keys.SPACE)
        time.sleep(0.1)

    def get_state(self):
        """截图并灰度化"""
        screenshot = self.driver.get_screenshot_as_png()
        img = Image.open(bytes(screenshot))
        img = img.convert('L')  # 灰度
        img = img.resize(STATE_SIZE)
        return np.array(img) / 255.0

    def jump(self):
        self.body.send_keys(Keys.SPACE)

    def step(self, action):
        """执行动作，0=不跳，1=跳"""
        if action == 1:
            self.jump()
        next_state = self.get_state()
        reward = 1.0  # 每帧奖励 1
        done = self.is_dead()
        return next_state, reward, done

    def is_dead(self):
        """简单像素判断死亡（可优化）"""
        # 可根据屏幕像素或游戏变量判断，这里示意
        return False  # 先用 False，后面可加入检测

    def close(self):
        self.driver.quit()

# ------------------- ε-greedy 策略 -------------------
def select_action(state, eps, policy_net, n_actions):
    if random.random() < eps:
        return random.randint(0, n_actions-1)
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = policy_net(state_tensor)
            return int(torch.argmax(q_values))

# ------------------- 主训练 -------------------
env = DinoEnv()
n_actions = 2  # 0=不跳，1=跳

policy_net = DQN(STATE_SIZE[0], STATE_SIZE[1], n_actions)
target_net = DQN(STATE_SIZE[0], STATE_SIZE[1], n_actions)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_SIZE)

epsilon = EPS_START
rewards_history = []

for episode in range(1, NUM_EPISODES+1):
    state = env.get_state()
    total_reward = 0

    for step in range(500):
        action = select_action(state, epsilon, policy_net, n_actions)
        next_state, reward, done = env.step(action)

        memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(memory) > BATCH_SIZE:
            s, a, r, s_, done_batch = memory.sample(BATCH_SIZE)
            q_values = policy_net(s)
            q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
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
    epsilon = max(EPS_END, epsilon * EPS_DECAY)

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if episode % 10 == 0:
        avg_reward = np.mean(rewards_history[-10:])
        print(f"Episode {episode}, avg reward={avg_reward:.2f}, epsilon={epsilon:.2f}")

env.close()
print("训练完成！")
