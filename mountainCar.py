import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import gym
from torch.autograd import Variable
import random
from collections import namedtuple

from tensorboardX import SummaryWriter

class DQN(nn.Module):
    def __init__(self, dueling=True):
        super().__init__()
        self.dueling = dueling
        self.fc1 = nn.Linear(2, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)
        if dueling:
            self.v = nn.Linear(512, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.dueling:
            v = self.v(x)
            a = self.fc3(x)
            q = v + a
        else:
            q = self.fc3(x)
        return q
        
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class Agent(object):
    def __init__(self, gamma=0.8, batch_size=128):
        self.target_Q = DQN()
        self.Q = DQN()
        self.gamma = gamma
        self.batch_size = 128
        hard_update(self.target_Q, self.Q)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=0.00001)
    
    def act(self, x, epsilon=0.1):
        p = random.uniform(0, 1)
        
        # Return random action with probability epsilon
        if p < epsilon :
            action = random.randint(0, 1)
            return action

        Q_sa = self.Q(x.data)
        argmax = Q_sa.data.numpy().argmax()
        return argmax

        
    
    def backward(self, transitions):
        batch = Transition(*zip(*transitions))

        state = Variable(torch.cat(batch.state))
        action = Variable(torch.from_numpy(np.array(batch.action)))
        next_state = Variable(torch.cat(batch.next_state))
        reward = Variable(torch.cat(batch.reward))
        done = Variable(torch.from_numpy(np.array(batch.done)))

        Q_sa = self.Q(next_state).detach()
        target = self.target_Q(next_state).detach() 
        _, argmax = Q_sa.max(dim=1, keepdim=True)
        target = target.gather(1, argmax)

        currentQvalues = self.Q(state).gather(1,action.unsqueeze(1)).squeeze()
        y = (reward.unsqueeze(1) + self.gamma * (target * (1-done.unsqueeze(1)))).squeeze()

        loss = F.smooth_l1_loss(currentQvalues,y)
        self.Loss = loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        soft_update(self.target_Q, self.Q, 0.995)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Writer to log data to tensorboard
writer = SummaryWriter('runs')

nb_epoch = 2000
env = gym.make('MountainCar-v0')

epsilon = 1
agent = Agent()
memory = ReplayMemory(100000)
batch_size = 128

for i in range(nb_epoch):
    obs = env.reset()
    done = False
    total_reward = 0
    epsilon *= 0.99
    action_histo = np.zeros(2)
    while not done:
        if i > nb_epoch-50:
            env.render()
        epsilon = max(epsilon, 0.1)

        obs_input = Variable(torch.from_numpy(obs).type(torch.FloatTensor))
        action = agent.act(obs_input, epsilon)

        next_obs, reward, done, _ = env.step(action)
        memory.push(obs_input.data.view(1,-1), action, 
                    torch.from_numpy(next_obs).type(torch.FloatTensor).view(1,-1), torch.Tensor([reward]),
                   torch.Tensor([done]))
        
        action_histo[action] += 1
        obs = next_obs
        total_reward += reward

        if reward == 1:
            print("WON AT EPISODE : ", i)
    writer.add_scalar('data/epsilon', epsilon, i)
    writer.add_scalar('data/total_reward', total_reward, i)
    writer.add_scalar('data/memory', memory.__len__(), i)
    writer.add_scalars('data/actions', {'0':action_histo[0], '1':action_histo[1]}, i)
    print("Episode : ", i)
    if memory.__len__() > 10000:
        batch = memory.sample(batch_size)
        agent.backward(batch)
        writer.add_scalar('data/loss', agent.Loss, i)

