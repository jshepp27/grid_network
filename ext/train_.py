import numpy as np
import environment_
from agent_ import Agent, DQN
import pickle
import utils
import json
import random

# LEARNING PARAMS
LEARNING_RATE = 0.001
N_ACTIONS = 3
INPUT_DIMS = 6
EPSILON = 1.00
GAMMA = 0.95
EPS_MIN = 0.01
N_GAMES = 1000
DECAY = 0.01
BUFFER_SIZE = 1000
BATCH_SIZE = 62
HORIZON = 100

# ENVIRONMENT PARAMS
CONSERVATISM = 0.55
RISK_THRESHOLD = 0.70
FAILURE_THRESHOLD = 0.90

grid = json.loads(open("../data/uk_grid.json", "r").read())

# TRAIN
if __name__ == "__main__":
    env = environment_ext.TransmissionAsset()
    scores = []
    eps_history = []
    eps_av = []
    total_steps = []

    obs, _, done, info = env.reset()
    n_games = N_GAMES

    agent = Agent(
        input_dims=obs.shape,
        n_actions=env.action_spaces.n,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        epsilon=EPSILON,
        eps_dec=DECAY,
        eps_min=EPS_MIN,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE
        )

    for _ in range(n_games):
        substation = random.sample(grid["substations"], k=1)[0]

        env = environment_ext.TransmissionAsset(
            substation=substation,
            train_mode=True,
            conservatism=CONSERVATISM,
            risk_threshold=RISK_THRESHOLD,
            failure_threshold=FAILURE_THRESHOLD
        )

        print(env.name)
        obs, reward, done, info = env.reset()

        score = 0
        steps = 0
        max_steps = HORIZON
        done = False

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
        total_steps.append(steps)
        avg_score = np.mean(scores)
        avg_steps = np.mean(total_steps)
        eps_history.append(agent.epsilon)
        print('episode ', _, 'score %.1f avg score %.1f, avg steps %.2f, epsilon %.2f, steps %.1f' %
                  (score, avg_score, avg_steps, agent.epsilon, steps))

    # STORE Agent
    with open("../models/trained_agent_.pkl", "wb") as f:
        pickle.dump(agent, f)

    x_axis = [i+1 for i in range(n_games)]

    filename = 'plots/training_asset_rehabilitation.png'
    title = "Learning Curve: Agent Rehab Training Performance"
    utils.create_plot(x_axis, scores, filename=filename, name=title)
