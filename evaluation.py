import pickle
import numpy as np
import utils
import environment
from agent import Agent, Random_Agent, Expert, ReplayBuffer, DQN
import matplotlib.pyplot as plt

class Evaluate:
    def __init__(self, env, agent, max_steps, run, random_agent=False, trials=100):
        self.env = env
        self.random_agent = random_agent
        self.agent = agent
        self.trials = trials
        self.max_steps = max_steps

        self.performance = []
        self.eps_history = []
        self.run = run

    def evaluate(self):
        obs, _, done, info = self.env.reset()

        for _ in range(self.trials):
            score = 0
            steps = 0
            obs, reward, done, info = self.env.reset()

            over = False
            while ((not over) and (steps < self.max_steps)):
                steps += 1
                action = self.agent.choose_action(obs, info["step"])

                obs_, reward, done, info = self.env.step(action)

                score += reward

                if done:
                    over = True

                obs = obs_
                #print(self.env.cost_history)

            self.performance.append(score)
            avg_perf = np.mean(self.performance)

            if self.random_agent == False:
                self.eps_history.append(self.agent.epsilon)
                print('episode ', _, 'score %.1f avg score %.1f epsilon %.2f, steps %.1f' %
                                   (score, avg_perf, self.agent.epsilon, steps))

            else:
                print('episode ', _, 'score %.1f avg score %.1f, steps %.1f' %
                      (score, avg_perf, steps))

            self.env.render(self.run)


with open("models/trained_agent.pkl", "rb") as f:
    AGENT = pickle.load(f)

TRIALS = 1
HORIZON = 200

# ENV = environment.TransmissionAsset(max_steps=HORIZON, train_mode=False)
# evaluation = Evaluate(ENV, AGENT, run="agent rehab", max_steps=HORIZON, trials=TRIALS)
# print(">> TRAINED AGENT")
# print("\n")
# evaluation.evaluate()
# plt.close()

# ENV = environment.TransmissionAsset(max_steps=HORIZON, train_mode=False)
# RAND_AGENT = Random_Agent(ENV.action_spaces.n)
# rand_evaluation = Evaluate(ENV, RAND_AGENT, run="agent random", max_steps=HORIZON, random_agent=True, trials=TRIALS)
# print(">> RANDOM AGENT")
# print("\n")
# rand_evaluation.evaluate()

ENV = environment.TransmissionAsset(max_steps=HORIZON, train_mode=False)
EXPERT = Expert()
expert_evaluation = Evaluate(ENV, EXPERT, run="agent greybeard", max_steps=HORIZON, random_agent=True, trials=TRIALS)
print(">> GREYBEARD AGENT")
print("\n")
expert_evaluation.evaluate()


