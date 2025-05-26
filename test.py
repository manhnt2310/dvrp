import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Đọc file CSV
file_path = "F:/KLTN/100_Customer/h100c101.csv"
df = pd.read_csv(file_path)

# Chuyển đổi dữ liệu thành numpy array
locations = df[['x', 'y']].values
demand = df['demand'].values
open_time = df['open'].values
close_time = df['close'].values
service_time = df['servicetime'].values
arrival_time = df['time'].values

# Chuẩn hóa tọa độ (nếu cần)
locations = locations / np.max(locations, axis=0)

import gym
from gym import spaces

class DVRPEnv(gym.Env):
    def __init__(self, df, max_capacity=1300):
        super(DVRPEnv, self).__init__()

        self.df = df
        self.max_capacity = max_capacity
        self.current_time = 0
        self.vehicle_capacity = max_capacity
        self.current_position = np.array([0, 0])  # Xe xuất phát từ (0,0)
        
        # Định nghĩa không gian hành động và trạng thái
        self.action_space = spaces.Discrete(len(df))  # Chọn 1 khách hàng để phục vụ
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(df), 5), dtype=np.float32)

        self.reset()

    def reset(self):
        self.current_time = 0
        self.vehicle_capacity = self.max_capacity
        self.current_position = np.array([0, 0])
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        return np.hstack((self.df[['x', 'y', 'demand', 'open', 'close']].values, 
                          np.array([self.current_time] * len(self.df)).reshape(-1, 1)))

    def step(self, action):
        selected = self.df.iloc[action]
        distance = np.linalg.norm(self.current_position - np.array([selected['x'], selected['y']]))
        
        if self.vehicle_capacity < selected['demand'] or self.current_time > selected['close']:
            reward = -100  # Phạt nếu không thể phục vụ khách hàng
        else:
            self.vehicle_capacity -= selected['demand']
            self.current_position = np.array([selected['x'], selected['y']])
            self.current_time += selected['servicetime'] + distance
            reward = -distance  # Thưởng dựa trên quãng đường ngắn nhất
        
        done = self.vehicle_capacity <= 0 or all(self.df['demand'] == 0)
        return self._get_observation(), reward, done, {}

env = DVRPEnv(df)

import torch.nn as nn

class MARDAMModel(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, output_dim=10):  # input_dim=7
        super(MARDAMModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Ánh xạ từ 7 → 128
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MARDAMModel()

import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for episode in range(500):  # Số lần huấn luyện
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_values = model(state_tensor).squeeze()
        action = torch.argmax(action_values).item()
        
        next_state, reward, done, _ = env.step(action)
        
        loss = loss_fn(action_values[action], torch.tensor(reward, dtype=torch.float32))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state = next_state
        total_reward += reward

    print(f"Episode {episode}, Total Reward: {total_reward}")
