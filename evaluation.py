
# Construct Environment
# Loop Environment
# Store Performance, Evaluation

import environment

import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
TODOs:
- Implement dill vs pickle for Agent persistence
"""

class Evaluate:
    def __init__(self, env, agent, max_steps, random_agent=False, trials=1000):
        self.env = env
        self.random_agent = random_agent
        self.agent = agent
        self.trials = trials
        self.max_steps = max_steps

        self.performance = []
        self.eps_history = []

    def evaluate(self):
        obs, _, done, info = self.env.reset()
        # self.agent = agent.Agent(lr=0.0001,
        #                          input_dims=obs.shape,
        #                          n_actions=self.env.action_spaces.n,
        #                          buffer_size=500,
        #                          batch_size=64)

        for _ in range(self.trials):
            score = 0
            steps = 0
            obs, reward, done, info = self.env.reset()

            over = False
            while ((not over) and (steps < self.max_steps)):
                steps += 1
                action = self.agent.choose_action(obs)
                obs_, reward, done, info = self.env.step(action)
                score += reward


                if done:
                    over = True

                obs = obs_


            self.performance.append(score)
            avg_perf = np.mean(self.performance)
            if self.random_agent == False:

                self.eps_history.append(self.agent.epsilon)
                print('episode ', _, 'score %.1f avg score %.1f epsilon %.2f, steps %.1f' %
                                   (score, avg_perf, self.agent.epsilon, steps))

            else:
                print('episode ', _, 'score %.1f avg score %.1f, steps %.1f' %
                      (score, avg_perf, steps))

            self.env.render()
            plt.show()

with open("trained_agent.pkl", "rb") as f:
    AGENT = pickle.load(f)

MAX_STEPS = 250
TRIALS = 1000
ENV = environment.TransmissionAsset(max_steps=MAX_STEPS)

evaluation = Evaluate(ENV, AGENT, MAX_STEPS, TRIALS)
print(">> TRAINED AGENT")
print("\n")
evaluation.evaluate()

x_axis = [i+1 for i in range(evaluation.trials)]
performance = evaluation.performance
#plot = utils.create_plot(x_axis, performance, filename="plots/evaluation_asset_rehabilitation.png")

#ENV = environment.TransmissionAsset()
#RAND_AGENT = Random_Agent(ENV.action_spaces.n)

#rand_evaluation = Evaluate(ENV, RAND_AGENT, MAX_STEPS, random_agent=True, trials=TRIALS)

# print(">> RANDOM AGENT")
# print("\n")
# rand_evaluation.evaluate()
#
# x_axis = [i+1 for i in range(rand_evaluation.trials)]
# performance = rand_evaluation.performance
# rand_plot = utils.create_plot(x_axis, performance, filename="plots/rand_evaluation_asset_rehabilitation.png")