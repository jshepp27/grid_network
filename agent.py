import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import numpy as np
import pickle

import environment
import random

from data.archive import utils

np.set_printoptions(precision=8, suppress=True)

# PARAMS
LEARNING_RATE = 0.001
N_ACTIONS = 3
INPUT_DIMS = 6
EPSILON = 1.00
GAMMA = 0.09
DECAY = 0.0001
EPS_MIN = 0.01
BUFFER_SIZE = 1000
BATCH_SIZE = 64

"""
TODOs:
- Agent Evaluation
-- Average over Multiple Episodes 
- Draw-Model Matrix Dimensions (exp. GPT Output) [DONE]

- Store Agent Model
- Scale state, reward, target values to work with Network
- Implement Sigmoid Activation
- Implement Experience Replay
- Argparse to parameterize training and eval.
 
"""

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
                 batch_size=BATCH_SIZE):

        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.exp_replay = ReplayBuffer(buffer_size=self.buffer_size)

        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [_ for _ in range(self.n_actions)]

        self.Q = DQN(self.lr, self.n_actions, self.input_dims)

    def choose_action(self, observation):
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

        actions_tensor = T.tensor(actions_taken, dtype=T.long).to(self.Q.device)
        q_pred = self.Q.forward(states).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        q_next = self.Q.forward(states_).max()

        q_target = rewards + self.gamma * q_next

        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()

class Random_Agent:
    def __init__(self, n_actions):
        self.action_space = n_actions
        self.epsilon = None
    def choose_action(self, observation):
        action = np.random.choice(self.action_space)

        return action

# TRAIN
if __name__ == "__main__":
    env = environment.TransmissionAsset()
    scores = []
    eps_history = []
    eps_av = []

    obs, _, done, info = env.reset()
    n_games = 5000
    print(obs)

    agent = Agent(lr=LEARNING_RATE, input_dims=obs.shape, n_actions=env.action_spaces.n, buffer_size=BUFFER_SIZE, batch_size=64)
    #print(agent.Q)

    for _ in range(n_games):
        score = 0
        steps = 0
        max_steps = 200
        done = False
        obs, reward, done, info = env.reset()

        over = False
        while ((not over) and (steps < max_steps)):
            steps += 1
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward

            # Increment Experience
            agent.exp_replay.add((obs, action, reward, obs_))
            agent.learn()

            obs = obs_

            # TODO: Improve Implementation
            if done:
                over = True
                # TODO: How to pass back Episode Success

        scores.append(score)
        avg_score = np.mean(scores)
        eps_history.append(agent.epsilon)
        print('episode ', _, 'score %.1f avg score %.1f epsilon %.2f, steps %.1f' %
                  (score, avg_score, agent.epsilon, steps))

    # STORE Agent
    with open("trained_agent.pkl", "wb") as f:
        pickle.dump(agent, f)

    filename = 'plots/asset_rehabilitation_dqn.png'
    x_axis = [i+1 for i in range(n_games)]

    filename = 'plots/training_asset_rehabilitation.png'
    utils.create_plot(x_axis, scores, filename=filename)
