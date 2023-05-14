import pickle
import numpy as np
import environment_
from agent_ import Agent, Random_Agent, Expert, ReplayBuffer, DQN
import string
import pandas as pd
import matplotlib.pyplot as plt

class Evaluate:
    def __init__(self, env, agent, max_steps, run, random_agent=False, trials=100, animate=True):
        self.env = env
        self.random_agent = random_agent
        self.agent = agent
        self.trials = trials
        self.max_steps = max_steps

        self.performance = []
        self.eps_history = []
        self.run = run
        self.animate = animate

        self.reliability = []
        self.intervention_cost = []

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

            self.reliability.append(1 - self.env.risk[-1])
            self.intervention_cost.append(sum(self.env.cost_history))

            if self.random_agent == False:
                self.eps_history.append(self.agent.epsilon)
                print('episode ', _, 'score %.1f avg score %.1f epsilon %.2f, steps %.1f' %
                                   (score, avg_perf, self.agent.epsilon, steps))

            else:
                print('episode ', _, 'score %.1f avg score %.1f, steps %.1f' %
                      (score, avg_perf, steps))

            if self.animate:
                self.env.render(self.run)


with open("../models/trained_agent_.pkl", "rb") as f:
    AGENT = pickle.load(f)

runs = {
    0: {
        "name": "agent random",
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
TRIALS = 100
HORIZON = 100

# ENVIRONMENT PARAMS
CONSERVATISM = 0.55
RISK_THRESHOLD = 0.70
FAILURE_THRESHOLD = 0.90

results = {
    "trials": TRIALS,
    "agent random": {
        "av_intervention_cost": 0,
        "av_reliability": 0
    },
    "expert greybeard": {
        "av_intervention_cost": 0,
        "av_reliability": 0
    },
    "agent rehab": {
        "av_intervention_cost": 0,
        "av_reliability": 0
    }
}


random_agent = False
animate = False

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
        trials=TRIALS,
        animate=animate
    )

    print(f"EVALUATING: {name.capitalize()}")
    EVALUATION.evaluate()

    intervention_cost = np.array(EVALUATION.intervention_cost)
    reliability = np.array(EVALUATION.reliability)

    av_intervention_cost = np.mean(intervention_cost)
    av_reliability = np.mean(reliability)

    results[name]["av_intervention_cost"] = av_intervention_cost
    results[name]["av_reliability"] = av_reliability


### OUTPUT RESULTS ###

results_df = pd.DataFrame(results)
results_df.to_csv("plots/results.csv")

agents = list(results.keys())[1:]  # Exclude 'trials' key
av_intervention_cost = [results[agent]['av_intervention_cost'] for agent in agents]
av_reliability = [results[agent]['av_reliability'] for agent in agents]

fig, ax1 = plt.subplots()
x = np.arange(len(agents))
bar_width = 0.35

# Plotting the average intervention cost
ax1.bar(x - bar_width/2, av_intervention_cost, width=bar_width, color='b', alpha=0.7, label='Average Intervention Cost')
ax1.set_xlabel('Agents')
ax1.set_ylabel('Average Intervention Cost')
ax1.tick_params(axis='y', labelcolor='b')

# Create a second subplot for average reliability
ax2 = ax1.twinx()
ax2.bar(x + bar_width/2, av_reliability, width=bar_width, color='r', alpha=0.7, label='Average Reliability')
ax2.set_ylabel('Average Reliability')
ax2.tick_params(axis='y')

# Adjust the plot settings for each axis
ax1.set_xticks(x)
ax1.set_xticklabels(agents)
ax1.set_title(f"Results over {results['trials']} Trials")

# Show legends for each axis
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig("evaluation_results")
plt.show()