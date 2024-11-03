from pettingzoo.mpe import simple_adversary_v3
import numpy as np
import torch
import torch.nn as nn
import os
import time
from maddpg_agent_v1 import Agent

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using device:{device}")


NUM_EPISODE = 10000
NUM_STEP = 200
MEMORY_SIZE = 100000
BATCH_SIZE = 512
TARGET_UPDATE_INTERVAL = 100
LR_ACTOR = 0.01
LR_CRITIC = 0.01
HIDDEN_DIM = 64
GAMMA = 0.99
TAU = 0.01


# 1 Initialize the agents
env = simple_adversary_v3.parallel_env(N=2, max_cycles=25, continuous_actions=True)
multi_obs, infos = env.reset()
NUM_AGENT = env.num_agents
agent_name_list = env.agents
# print(multi_obs)
# print(agent_name_list)
# for agent_i,agent_name in enumerate(agent_name_list):
#     print(agent_i)

scenario = "simple_adversary_v3"
current_path = os.path.dirname(os.path.realpath(__file__))
agent_path = current_path + "/models/" + scenario + '/'
timestamp = time.strftime("%Y%m%d%H%M%S")
print(timestamp)

def multi_obs_to_state(multi_obs):
    state = np.array([])
    for agent_obs in multi_obs.values():
        state = np.concatenate([state,agent_obs])
    return state


# 1.1 Get obs_dim
obs_dim = []
for agent_obs in multi_obs.values():
    obs_dim.append(agent_obs.shape[0])
state_dim = sum(obs_dim)     # 状态空间维度等于各个agent观测空间维度和

# 1.2 Get action_dim
action_dim = []
for agent_name in agent_name_list:
    action_dim.append(env.action_space(agent_name).sample().shape[0])


agents = []
for agent_i in range(NUM_AGENT):
    print(f"Initializing agent {agent_i}...")
    agent = Agent(memo_size=MEMORY_SIZE, obs_dim=obs_dim[agent_i], state_dim=state_dim, n_agent=NUM_AGENT,
                  action_dim=action_dim[agent_i], alpha=LR_ACTOR, beta=LR_CRITIC, fc1_dims=HIDDEN_DIM,
                  fc2_dims=HIDDEN_DIM, gamma=GAMMA, tau=TAU, batch_size=BATCH_SIZE)
    agents.append(agent)    # 创建智能体实例列表


# 2 Main training loop
for episode_i in range(NUM_EPISODE):
    multi_obs, infos = env.reset()
    episode_reward = 0
    multi_done = {agent_name:False for agent_name in agent_name_list}

    for step_i in range(NUM_STEP):
        total_step = episode_i * NUM_STEP + step_i

        # 2.1 Collect actions from all agents
        multi_actions = {}
        for agent_i,agent_name in enumerate(agent_name_list):      # 同时获取列表中每个元素的索引和元素值
            agent = agents[agent_i]
            single_obs = multi_obs[agent_name]
            single_action = agent.get_action(single_obs)
            multi_actions[agent_name] = single_action

        # 2.2 Execute action
        multi_next_obs,multi_reward,multi_done,multi_truncation,infos = env.step(multi_actions)
        state = multi_obs_to_state(multi_obs)      # 将各智能体的局部观测转化为全局状态
        next_state = multi_obs_to_state(multi_next_obs)
        if step_i >= NUM_STEP - 1:
            multi_done = {agent_name:True for agent_name in agent_name_list}

        # 2.3 Store memory
        for agent_i,agent_name in enumerate(agent_name_list):
            agent = agents[agent_i]
            single_obs = multi_obs[agent_name]
            single_next_obs = multi_next_obs[agent_name]
            single_action = multi_actions[agent_name]
            single_reward = multi_reward[agent_name]
            single_done = multi_done[agent_name]
            agent.replay_buffer.add_memo(single_obs,single_next_obs,state,next_state,
                                         single_action,single_reward,single_done)

        # 2.4 Update brain every fixed steps
        multi_batch_obses = []
        multi_batch_next_obses = []
        multi_batch_states = []
        multi_batch_next_states = []
        multi_batch_actions = []
        multi_batch_next_actions = []
        multi_batch_online_actions = []
        multi_batch_rewards = []
        multi_batch_dones = []


        # 2.4.1 Sample a batch of memories
        current_memo_size = min(MEMORY_SIZE, total_step + 1)
        if current_memo_size < BATCH_SIZE:
            batch_idx = range(0, current_memo_size)
        else:
            batch_idx = np.random.choice(current_memo_size, BATCH_SIZE)

        for agent_i in range(NUM_AGENT):
            agent = agents[agent_i]
            batch_obses, batch_next_obses, batch_states, batch_next_state,\
              batch_actions, batch_rewards, batch_dones = agent.replay_buffer.sample(batch_idx)

            # Single + batch
            batch_obses_tensor = torch.tensor(batch_obses,dtype=torch.float).to(device)
            batch_next_obses_tensor = torch.tensor(batch_next_obses,dtype=torch.float).to(device)
            batch_states_tensor = torch.tensor(batch_states,dtype=torch.float).to(device)
            batch_next_state_tensor = torch.tensor(batch_next_state,dtype=torch.float).to(device)
            batch_actions_tensor = torch.tensor(batch_actions,dtype=torch.float).to(device)
            batch_rewards_tensor = torch.tensor(batch_rewards,dtype=torch.float).to(device)
            batch_dones_tensor = torch.tensor(batch_dones,dtype=torch.float).to(device)

            # Multiple + batch
            multi_batch_obses.append(batch_obses_tensor)
            multi_batch_next_obses.append(batch_next_obses_tensor)
            multi_batch_states.append(batch_states_tensor)
            multi_batch_next_states.append(batch_next_state_tensor)
            multi_batch_actions.append(batch_actions_tensor)


            single_batch_next_actions = agent.target_actor.forward(batch_next_obses_tensor)
            multi_batch_next_actions.append(single_batch_next_actions)

            single_batch_online_action = agent.actor.forward(batch_obses_tensor)
            multi_batch_online_actions.append(single_batch_online_action)


            multi_batch_rewards.append(batch_rewards_tensor)
            multi_batch_dones.append(batch_dones_tensor)

        multi_batch_actions_tensor = torch.cat(multi_batch_actions, dim=1).to(device)
        multi_batch_next_actions_tensor = torch.cat(multi_batch_next_actions, dim=1).to(device)
        multi_batch_online_actions_tensor = torch.cat(multi_batch_online_actions, dim=1).to(device)


        # Update critic and actor
        if (total_step + 1) % TARGET_UPDATE_INTERVAL == 0:

            for agent_i in range(NUM_AGENT):
                agent = agents[agent_i]

                batch_obses_tensor = multi_batch_obses[agent_i]
                batch_states_tensor = multi_batch_states[agent_i]
                batch_next_states_tensor = multi_batch_next_states[agent_i]
                batch_rewards_tensor = multi_batch_rewards[agent_i]
                batch_dones_tensor = multi_batch_dones[agent_i]
                batch_actions_tensor = multi_batch_actions[agent_i]

                # target critic
                critic_target_q = agent.target_critic.forward(batch_states_tensor,
                                                              multi_batch_online_actions_tensor.detach())
                y = (batch_rewards_tensor + (1 - batch_dones_tensor) * agent.gamma * critic_target_q).flatten()

                critic_q = agent.critic.forward(batch_states_tensor, multi_batch_actions_tensor).flatten()

                # Update critic
                critic_loss = nn.MSELoss()(y, critic_q)
                agent.critic.optimizer.zero_grad()
                critic_loss.backward()
                agent.critic.optimizer.step()

                # Update actor
                actor_loss = agent.critic.forward(batch_states_tensor,
                                                  multi_batch_online_actions_tensor.detach()).flatten()
                actor_loss = -torch.mean(actor_loss)
                agent.actor.optimizer.zero_grad()
                critic_loss.backward()
                agent.actor.optimizer.step()

                # Update target critic
                for target_param, param in zip(agent.target_actor.parameters(), agent.critic.parameters()):
                    target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param.data)

                # Update target actor
                for target_param, param in zip(agent.target_actor.parameters(), agent.actor.parameters()):
                    target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param.data)

        multi_obs = multi_next_obs
        episode_reward += sum([single_reward for single_reward in multi_reward.values()])
        print(f"Episode reward:{episode_reward}")



    # 3 Render the env
    if (episode_i + 1) % 10 == 0:
        env = simple_adversary_v3.parallel_env(N=2,
                                               max_cycles=NUM_STEP,
                                               continuous_actions=True,
                                               render_mode="human")

        for test_epi_i in range(10):
            multi_obs, infos = env.reset()
            for step_i in range(NUM_STEP):
                multi_actions = {}
                for agent_i, agent_name in enumerate(agent_name_list):  # 同时获取列表中每个元素的索引和元素值
                    agent = agents[agent_i]
                    single_obs = multi_obs[agent_name]
                    single_action = agent.get_action(single_obs)
                    multi_actions[agent_name] = single_action
                multi_next_obs, multi_reward, multi_done, multi_truncation, infos = env.step(multi_actions)
                multi_obs = multi_next_obs


    # 4 Save the agents' models
    if episode_i == 0:
        highest_reward = episode_reward
    if episode_reward > highest_reward:
        highest_reward = episode_reward
        print(f"Highest reward updated at episode {episode_i}: {round(highest_reward, 2)}")
        for agent_i in range(NUM_AGENT):
            agent = agents[agent_i]
            flag = os.path.exists(agent_path)
            if not flag:
                os.makedirs(agent_path)
            torch.save(agent.actor.state_dict(), f"{agent_path}" + f"agent_{agent_i}_actor_{scenario}_{timestamp}.pth")

env.close()