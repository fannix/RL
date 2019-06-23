import gym
import torch
from torch import nn, optim
from collections import namedtuple
import torch.nn.functional as F
import random
import numpy as np
from torch.distributions import Categorical
# from torch.utils.data import Dataset, DataLoader

class Policy(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, output_size)
    
    def forward(self, X):
        hidden = F.leaky_relu(self.input(X))
        output = F.leaky_relu(self.hidden(hidden))
        return F.log_softmax(output)

env = gym.make("CartPole-v0")

def get_episodes_on_policy(policy):
    state = env.reset()
    done = False

    episode_reward = 0
    episode_len = 0

    action_score_li = []
    reward_li = []

    total_reward = 0
    num_episode = 0
    while not (done and len(action_score_li) >= 5000):
        predict = policy(torch.FloatTensor(state))
        # sample from policy
        action = Categorical(logits=predict).sample().item()
        action_score = predict[action]

        # the list will hold the tensor produced by the policy network
        # it should be fine since the network won't change inside the epoch
        action_score_li.append(action_score)

        state, reward, done, _ = env.step(action)
        episode_reward += reward
        total_reward += reward
        episode_len += 1

        if done:
            num_episode += 1
            reward_li += [episode_reward] * episode_len
            state = env.reset()
            episode_reward = 0
            episode_len = 0
    
    assert len(action_score_li) == len(reward_li), "{} should be the same as {}".format(
        len(action_score_li), len(reward_li))


    return action_score_li, reward_li, total_reward, num_episode



policy = Policy(4, 32, env.action_space.n)
optimizer = optim.Adam(policy.parameters())

for epoch in range(50):
    action_score_li, reward_li, total_reward, num_episode = get_episodes_on_policy(policy)
    print("epoch %d \t avg_reward %.3f" % (epoch, total_reward / num_episode))

    for i, action_score in enumerate(action_score_li):
        optimizer.zero_grad()
        reward = reward_li[i]
        loss = - action_score * reward
        loss.backward()
        optimizer.step()

        if i % 100 == 99:
            print("loss", loss.item())
