import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import numpy as np
import environment
import matplotlib.pyplot as plt

np.set_printoptions(precision=8, suppress=True)

# PARAMS
LEARNING_RATE = 0.0001
N_ACTIONS = 3
INPUT_DIMS = 6
EPSILON = 1.00
GAMMA = 0.09
DECAY = 0.03
EPS_MIN = 0.01

"""
TODOs:
- Agent Evaluation
-- Average over Multiple Episodes

- Store Agent Model
- Scale state, reward, target values to work with Network
- Implement Sigmoid Activation
- Implement Experience Replay
 
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

class Agent():
    def __init__(self,
                 input_dims=INPUT_DIMS,
                 n_actions=N_ACTIONS,
                 lr=LEARNING_RATE,
                 gamma=GAMMA,
                 epsilon=EPSILON,
                 eps_dec=DECAY,
                 eps_min=EPS_MIN):

        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [_ for _ in range(self.n_actions)]

        self.Q = DQN(self.lr, self.n_actions, self.input_dims)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(observation, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = T.argmax(actions)


        else:
            action = np.random.choice(self.action_space)

        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state, action, reward, state_):
        # TODO (?) Clear Optimizer
        self.Q.optimizer.zero_grad()

        states = T.tensor(state, dtype=T.float).to(self.Q.device)
        actions = T.tensor(action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.Q.device)

        # NOTE: Absolute Q-Values, not Actions. Thus, no Argmax.
        # TODO (?) [actions] controls correct batch size and shape
        q_pred = self.Q.forward(states)[actions]

        q_next = self.Q.forward(states_).max()

        q_target = rewards + self.gamma * q_next

        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()

def create_plot(x_axis, scores):
    fig, ax = plt.subplots()
    ax.plot(x_axis, scores, label="scores")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 100):(t + 1)])

    ax.plot(x_axis, running_avg, label="average")
    ax.set_title("Learning Curve DQN")
    ax.legend()
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Learning")
    return fig, ax

# TRAIN
if __name__ == "__main__":
    env = environment.TransmissionAsset()
    scores = []
    eps_history = []
    eps_av = []

    obs, _, done, info = env.reset()
    n_games = 500
    print(obs)

    #print(obs.shape)
    #print(env.action_spaces.n)

    agent = Agent(lr=0.0001, input_dims=obs.shape, n_actions=env.action_spaces.n)
    #print(agent.Q)

    for _ in range(n_games):
        score = 0
        steps = 0
        max_steps = 1000
        done = False
        obs, reward, done, info = env.reset()

        over = False
        while ((not over) and (steps < max_steps)):
            steps += 1
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward
            agent.learn(obs, action, reward, obs_)
            obs = obs_

            # TODO: Improve Implementation
            if done:
                if env.unmanaged_failure == False:
                    reward += 100
                else:
                    over = True

        scores.append(score)
        avg_score = np.mean(scores)
        eps_history.append(agent.epsilon)
        print('episode ', _, 'score %.1f avg score %.1f epsilon %.2f, steps %.1f' %
                  (score, avg_score, agent.epsilon, steps))


filename = 'asset_rehabilitation_dqn.png'
x_axis = [i+1 for i in range(n_games)]

# TODO: Implement Individual Episode Curves OR Multi-Training Curves, different Params
create_plot(x_axis, scores)

plt.savefig(filename)
plt.show()

# N = len(scores)
# running_avg = np.empty(N)
# for t in range(N):
#     running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])

# DUMB LOOP
# for _ in range(100):
#     state, reward, done, info = env.step(action=0)
#     if done:
#         print("ASSET FAILED")
#         print(info["probability_failure"])
#         break
#
#     else:
#         print(info)
#         print(state)
#         print(reward, done)
#         print("")
