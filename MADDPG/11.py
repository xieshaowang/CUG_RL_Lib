from pettingzoo.mpe import simple_adversary_v3
import numpy as np
import torch
import torch.nn as nn
import os
import time
import torch.nn.functional as F

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using device:{device}")


env = simple_adversary_v3.parallel_env(N=2, max_cycles=25, continuous_actions=True)
multi_obs, infos = env.reset()
NUM_AGENT = env.num_agents
agent_name_list = env.agents

# print(multi_obs)
# print(infos)
# print(NUM_AGENT)
# print(agent_name_list)
# for agent_i,agent_name in enumerate(agent_name_list):
#     print(agent_i)


# obs_dim = []
# for agent_obs in multi_obs.values():
#     obs_dim.append(agent_obs.shape[0])
#     print(agent_obs.shape)
#     print(agent_obs)
# state_dim = sum(obs_dim)
# print(multi_obs.values())
# print(obs_dim)
# print(state_dim)

# action_dim = []
# for agent_name in agent_name_list:
#     action_dim.append(env.action_space(agent_name).sample().shape[0])
#     print(env.action_space(agent_name).sample())


input_dims = 8
fc1_dims = 64
fc2_dims = 64
action_dims = 5
fc1 = nn.Linear(input_dims,fc1_dims).to(device)    # 定义第一个全连接层
fc2 = nn.Linear(fc1_dims,fc2_dims).to(device)       # 定义第二个全连接层
pi = nn.Linear(fc2_dims,action_dims).to(device)
obs = multi_obs[agent_name_list[0]]
single_obs = torch.tensor(data=obs, dtype=torch.float).unsqueeze(0).to(device)
print(obs)
print(single_obs)
x = F.relu(fc1(single_obs))   # relu是一个常用的非线性激活函数，用于引入非线性特征
x = F.relu(fc2(x))
mu = torch.softmax(pi(x), dim=1)
print(pi(x))
print(mu)









# for agent_i in range(NUM_AGENT):
#     print(agent_i)


# state = np.array([])
# for agent_obs in multi_obs.values():
#     state = np.concatenate([state,agent_obs])
# print(state)

# multi_done = {agent_name:False for agent_name in agent_name_list}
# print(multi_done)
# print(multi_done.values())

# for agent_i,agent_name in enumerate(agent_name_list):
#     print(agent_i)
#     print(agent_name)


# obs_cap = np.empty((10, 2))
# print(obs_cap)

# multi_done = {agent_name:False for agent_name in agent_name_list}
# print(multi_done)

# obs_cap = np.empty((10, 1))
# print(obs_cap)






# NUM_EPISODE = 10
# NUM_STEP = 10
# # 2 Main training loop
# for episode_i in range(NUM_EPISODE):
#     print(episode_i)


