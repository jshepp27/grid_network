import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import numpy as np
import pickle
import random

# PARAMS
LEARNING_RATE = 0.001
N_ACTIONS = 3
INPUT_DIMS = 6
EPSILON = 1.00
GAMMA = 0.95
EPS_MIN = 0.01
N_GAMES = 1000
DECAY = 0.0001
BUFFER_SIZE = 100000
BATCH_SIZE = 62

class DQN(nn.Module):
    def __init__(self, lr=LEARNING_RATE, n_actions=N_ACTIONS, input_dims=INPUT_DIMS):
        super(DQN, self).__init__()

        self.input_dims = input_dims
        self.fc1 = nn.Linear(*self.input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        #print("state shape before view:", state.shape)
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)

        return actions

# TODO: Consider 'game-over' state
class ReplayBuffer:
    def __init__(self, buffer_size=100):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, transition):
        experience = transition
        # MANAGE BUFFER AS A QUEUE FIFO
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)

        self.buffer.append(experience)
    def sample(self, batch_size=64):
        sampled_experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, states_ = [], [], [], []

        for exp in sampled_experiences:
            state, action, reward, state_ = exp
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            states_.append(state)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        states_ = np.array(states_)

        return states, actions, rewards, states_
class Agent():
    def __init__(self,
                 input_dims=INPUT_DIMS,
                 n_actions=N_ACTIONS,
                 lr=LEARNING_RATE,
                 gamma=GAMMA,
                 epsilon=EPSILON,
                 eps_dec=DECAY,
                 eps_min=EPS_MIN,
                 buffer_size=BUFFER_SIZE,
                 batch_size=BATCH_SIZE,
                 mode=None):

        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.exp_replay = ReplayBuffer(buffer_size=self.buffer_size)

        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [_ for _ in range(self.n_actions)]

        self.Q = DQN(self.lr, self.n_actions, self.input_dims)
        self.gamma = T.tensor(gamma, dtype=T.float32).to(self.Q.device)

    def choose_action(self, observation=None, time_step=None):
        if np.random.random() > self.epsilon:
            # EXPLORE
            state = T.tensor(observation, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = T.argmax(actions)

        else:
            # EXPLOIT
            action = np.random.choice(self.action_space)

        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        if len(self.exp_replay.buffer) < self.batch_size:
            return

        state, action, reward, state_ = self.exp_replay.sample(self.batch_size)

        self.Q.optimizer.zero_grad()

        states = T.tensor(state, dtype=T.float).to(self.Q.device)
        actions_taken = T.tensor(action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.Q.device)

        #actions_tensor = T.tensor(actions_taken, dtype=T.long).to(self.Q.device)
        actions_tensor = actions_taken.clone().detach().long().to(self.Q.device)
        q_pred = self.Q.forward(states).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        q_next = self.Q.forward(states_).max()

        #q_target = rewards + self.gamma * q_next.double()
        q_target = (rewards + self.gamma * q_next).float()

        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()

class Random_Agent:
    def __init__(self, action_space, mode=None):
        self.action_space = action_space
        self.epsilon = None

    def choose_action(self, observation, time_step=None):
        action = np.random.choice(self.action_space)

        return action

# Recurring Schedule 5-10
class Expert:
    def __init__(self, mode):
        self.T = 0
        self.mode = mode
        self.maintain = 1

    def choose_action(self, observation=None, time_step=None):
        """
        The human agent makes decisions based on simple heuristic rules.
        These might be based on the current state of the system (observation),
        or other factors depending on the specific context of your problem.
        """
        if time_step % self.mode == 0:
            action = self.maintain

        else:
            action = 0

        return action