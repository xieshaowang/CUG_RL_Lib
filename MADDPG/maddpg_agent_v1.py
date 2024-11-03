from typing import Union, Callable, Tuple, Any, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self,capacity, obs_dim, state_dim,
                 action_dim, batch_size):
        self.capacity = capacity
        self.obs_cap = np.empty((self.capacity, obs_dim))    # 创建一个指定形状的数组
        self.next_obs_cap = np.empty((self.capacity, obs_dim))
        self.state_cap = np.empty((self.capacity, state_dim))
        self.next_state_cap = np.empty((self.capacity, state_dim))
        self.action_cap = np.empty((self.capacity, action_dim))
        self.reward_cap = np.empty((self.capacity, 1))
        self.done_cap = np.empty((self.capacity, 1), dtype=bool)

        self.batch_size = batch_size
        self.current = 0


    def add_memo(self, obs, next_obs, state, next_state, action, reward, done):

        self.obs_cap[self.current] = obs
        self.next_obs_cap[self.current] = next_obs
        self.state_cap[self.current] = state
        self.next_state_cap[self.current] = next_state
        self.action_cap[self.current] = action
        self.reward_cap[self.current] = reward
        self.done_cap[self.current] = done

        self.current = (self.current + 1) % self.capacity

    def sample(self, idxes):
        obs = self.obs_cap[idxes]
        next_obs = self.next_obs_cap[idxes]
        state = self.state_cap[idxes]
        next_state = self.next_state_cap[idxes]
        action = self.action_cap[idxes]
        reward = self.reward_cap[idxes]
        done = self.done_cap[idxes]

        return obs, next_obs, state, next_state, action, reward, done


class Critic(nn.Module):
    def __init__(self, lr_critic, input_dims, fc1_dims, fc2_dims, n_agent, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(input_dims + n_agent * action_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_critic)

    def forward(self,state,action):
        x = torch.cat([state,action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))




class Actor(nn.Module):
    def __init__(self, lr_actor, input_dims, fc1_dims, fc2_dims, action_dims):
        super(Actor,self).__init__()

        self.fc1 = nn.Linear(input_dims,fc1_dims)     # 定义第一个全连接层
        self.fc2 = nn.Linear(fc1_dims,fc2_dims)       # 定义第二个全连接层
        self.p = nn.Linear(fc2_dims,action_dims)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_actor)

    def forward(self,obs):
        x = F.relu(self.fc1(obs))   # relu是一个常用的非线性激活函数，用于引入非线性特征
        x = F.relu(self.fc2(x))
        mu = torch.softmax(self.p(x), dim=1)
        return mu

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))


class Agent:
    def __init__(self,memo_size, obs_dim, state_dim, n_agent,action_dim,
                 alpha, beta, fc1_dims, fc2_dims, gamma, tau, batch_size):
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        self.actor = Actor(lr_actor=alpha, input_dims=obs_dim, fc1_dims=fc1_dims,
                                fc2_dims=fc2_dims, action_dims=action_dim).to(device)

        self.critic = Critic(lr_critic=beta, input_dims=state_dim, fc1_dims=fc1_dims,
                              fc2_dims=fc2_dims, n_agent=n_agent, action_dim=action_dim).to(device)

        self.target_actor = Actor(lr_actor=alpha, input_dims=obs_dim, fc1_dims=fc1_dims,
                                fc2_dims=fc2_dims, action_dims=action_dim).to(device)

        self.target_critic = Critic(lr_critic=beta, input_dims=state_dim, fc1_dims=fc1_dims,
                             fc2_dims=fc2_dims, n_agent=n_agent, action_dim=action_dim).to(device)

        self.replay_buffer = ReplayBuffer(capacity=memo_size, obs_dim=obs_dim, state_dim=state_dim,
                 action_dim=action_dim, batch_size=batch_size)

    def get_action(self,obs):
        single_obs = torch.tensor(data=obs, dtype=torch.float).unsqueeze(0).to(device)
        single_action = self.actor.forward(single_obs)
        noise = torch.randn(self.action_dim).to(device) * 0.2
        single_action = torch.clamp(input=single_action+noise, min=0.0, max=1.0)

        return single_action.detach().cpu().numpy()[0]

    def save_model(self,filename):
        self.actor.save_checkpoint(filename)
        self.target_actor.save_checkpoint(filename)
        self.critic.save_checkpoint(filename)
        self.target_critic.save_checkpoint(filename)

    def load_model(self,filename):
        self.actor.load_checkpoint(filename)
        self.target_actor.load_checkpoint(filename)
        self.critic.load_checkpoint(filename)
        self.target_critic.load_checkpoint(filename)