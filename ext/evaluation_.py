import pickle
import numpy as np
import environment_
from agent_ import Agent, Random_Agent, Expert, ReplayBuffer, DQN
import string

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
                action = self.agent.choose_action(obs, info["time_step"])

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


with open("../models/trained_agent_.pkl", "rb") as f:
    AGENT = pickle.load(f)

runs = {
    0: {
        "name": "random agent",
        "random_agent": True,
        "agent": Random_Agent(action_space=3)
    },
    1: {
        "name": "expert greybeard",
        "agent": Expert(mode=5)
    },
    2: {
        "name": "agent rehab",
        "agent": AGENT
    }
}

# EVALUATION PARAMS
TRIALS = 1
HORIZON = 100

# ENVIRONMENT PARAMS
CONSERVATISM = 0.55
RISK_THRESHOLD = 0.70
FAILURE_THRESHOLD = 0.90

random_agent = False
for _ in runs.keys():
    name = runs[_]["name"]
    random_agent = "random_agent" in runs[_].keys()

    AGENT = runs[_]["agent"]
    ENV = environment_.TransmissionAsset(
        max_steps=HORIZON,
        train_mode=False,
        conservatism=CONSERVATISM,
        risk_threshold=RISK_THRESHOLD,
        failure_threshold=FAILURE_THRESHOLD
    )

    print(ENV.action_spaces.n)

    run = string.capwords(name)
    EVALUATION = Evaluate(
        ENV,
        AGENT,
        run=run,
        max_steps=HORIZON,
        random_agent=True,
        trials=TRIALS
    )

    print(f"EVALUATING: {name.capitalize()}")
    EVALUATION.evaluate()


