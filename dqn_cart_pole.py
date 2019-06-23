import gym
import torch
from torch import nn, optim
from collections import namedtuple
import torch.nn.functional as F
import random
import numpy as np

Play = namedtuple(
    'Play', ['state', 'action', 'next_state', 'reward', 'done'])

class ReplayBuffer(object):

    def __init__(self):
        self.buffer = []

    def append(self, play):
        self.buffer.append(play)

    def size(self):
        return len(self.buffer)
    
    def sample(self):
        return random.choice(self.buffer)

env = gym.make("CartPole-v0")

# def get_sample(nsample):
#     state = env.reset()


#     while buffer.size() < nsample:
#         action = env.action_space.sample()
#         next_state, reward, done, info = env.step(action)

#         play = Play(state, action, next_state, reward, done)
#         buffer.append(play)

#         if done:
#             state = env.reset()
#         else:
#             state = next_state

buffer = ReplayBuffer()

# n_sample = 10
# test_X = torch.randn(n_sample, 4)
# test_y = torch.FloatTensor([random.uniform(0, 1) for _ in range(n_sample)])
# test_y = test_y.unsqueeze(1)


class DQN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, output_size)
    
    def forward(self, X):
        hidden = F.leaky_relu(self.input(X))
        output = F.leaky_relu(self.hidden(hidden))
        return output

loss = nn.SmoothL1Loss()
gamma = 0.8

def get_epsilon(epoch, total):
    return (total - epoch) / total 


dqn = DQN(4, 16, 2)
optimizer = optim.Adam(dqn.parameters())
for episode in range(10000):
    state = env.reset()

    total_epoch = 200
    for epoch in range(total_epoch):
        epsilon = get_epsilon(epoch, total_epoch)

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax(dqn(torch.FloatTensor(state))).item()

        next_state, reward, done, info = env.step(action)
        play = Play(state, action, next_state, reward, done)
        buffer.append(play)

        ### sampling
        sampled_play = buffer.sample()
        if sampled_play.done:
            target = sampled_play.reward
        else:
            target = sampled_play.reward + \
                gamma * torch.max(dqn(torch.FloatTensor(sampled_play.next_state))).item()
        
        predict = dqn(torch.FloatTensor(sampled_play.state))[sampled_play.action]
        
        optimizer.zero_grad()
        l = loss(target, predict)
        l.backward()
        optimizer.step()

        state = next_state

        if done:
            break

    print(epoch, l.item(), target, predict.data)

########### Test #########

def test():
    state = env.reset()
    done = False

    i = 0
    while not done:
        i += 1
        q = dqn(torch.FloatTensor(state))

        action = torch.argmax(q).item()
        print(i, action, q.data)

        state, reward, done, info = env.step(action)

for i in range(5):
    test()

    print()
