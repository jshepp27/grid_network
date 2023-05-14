import gym
from gym import spaces
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import clear_output
from matplotlib.ticker import FuncFormatter
import string

np.set_printoptions(precision=8, suppress=True)

# TODO: Evaluation Charts
# TODO: Renames: age_intrinsic
# TODO: Experiment with Voltage as an Absolute Value
# TODO: Fix Cost-Proportions

# TODO: Tune Reward-Training Results [DONE]
# TODO: Work Penalty [DONE]
# TODO: Simulation Blueprint [DONE]
# TODO: Implement Human-Agent [DONE]
# TODO: Run Evaluation w Visualisation [DONE]
# TODO: Implement Stochastic Failure [DONE]
# TODO: Refactor and Understand Class [DONE]
# TODO: Experiment with Reward Functions [DONE]

import json
import random

grid = json.loads(open("../data/uk_grid.json", "r").read())

impacts = grid["intervention_impacts"]
costs = grid["intervention_costs"]
voltage_types = grid["voltage_types"]
substation = random.sample(grid["substations"], k=1)[0]

CONSERVATISM = 0.55
RISK_THRESHOLD = 0.70
FAILURE_THRESHOLD = 0.90

class TransmissionAsset(gym.Env):
    def __init__(self,
                 impacts=impacts,
                 costs=costs,
                 voltage_types=voltage_types,
                 substation=substation,
                 max_steps=100,
                 visualise=False,
                 train_mode=True,
                 conservatism=CONSERVATISM,
                 risk_threshold=RISK_THRESHOLD,
                 failure_threshold=FAILURE_THRESHOLD):

        super(TransmissionAsset, self).__init__()

        self.visualise = visualise
        self.train_mode = train_mode
        self.max_steps = max_steps
        self.horizon = range(0, self.max_steps)
        self.conservatism = conservatism

        self.unmanaged_failure = False
        self.risk_threshold = risk_threshold
        self.failure_threshold = failure_threshold
        self.time_step = 0

        if self.visualise:
            self.fig, self.ax1 = plt.subplots()
            self.ax2 = self.ax1.twinx()
            self.ani = None

        self.substation = substation
        self.name = substation["name"]
        self.initial_age = substation["age"]
        self.voltage_types = voltage_types
        self.voltage = substation["voltage"]
        self.voltage_one_hot = np.zeros(3)

        self.failure_rate = self.substation["failure_rate"]

        self.impacts = impacts
        self.fix_lower = self.impacts["fix"]["lower_bound"]
        self.fix_upper = self.impacts["fix"]["upper_bound"]
        self.refurbish_upper = self.impacts["refurbish"]
        self.costs = costs

        self.c_max = max(self.costs.values())
        self.c_min = min(self.costs.values())
        self.c_nothing = self.costs["do_nothing"]
        self.c_fix = self.costs["fix"]
        self.c_refurbish = self.costs["refurbish"]
        self.c_replace = self.costs["replace"]
        self.c_penalty = self.costs["replace"]

        self.CONSTANT_POSITIVE_REWARD = 1.0
        self.INACTION_PENALTY = 0.1
        self.reward_min = 1
        self.reward_max = -1

        self.action_spaces = spaces.Discrete(3)
        self.observation_space = spaces.Dict({
            "age": spaces.Box(low=0, high=float("inf"), shape=(1,)),
            "voltage": spaces.Discrete(len(self.voltage_types)),
            "failure_rate": spaces.Box(low=0, high=1, shape=(1,)),
            "EOL": spaces.Box(low=0, high=1, shape=(1,)),
        })

        self.state = {}
        self.state_vector = np.zeros(len(self.observation_space.keys())).reshape(-1, 1)

        self.action_history = []
        self.action_costs = []

        self.age_max = max(self.horizon)
        self.age_min = min(self.horizon)

        self.cost_history = []
        self.reward_history = []
        self.risk = []

    def eol(self, age):
        return float(1 - math.exp(-self.failure_rate * age))

    def degrade(self, action):
        if action == 0:
            self.state["age"] += 1.0

        elif action == 1:
            age_reduction = self.state["age"] - np.random.uniform(self.fix_lower, self.fix_upper)
            self.state["age"] = round(max(self.state["age"] - age_reduction, 1))

        else:
            self.state["age"] = self.state["age"] - (self.refurbish_upper)*self.state["age"]

        self.state["EOL"] = self.eol(self.state["age"])

    def compute_reward(self, action):
        reward = -self.eol(age=self.state["age"])

        c_nothing = 0.05
        c_fix = 0.35
        c_refurbish = 0.7
        c_issue = 1

        if action == 0:
            self.cost_history.append(c_nothing)
            cost = c_nothing if self.state["EOL"] < self.risk_threshold else c_issue
            reward -= cost
        if action == 1:
            self.cost_history.append(c_fix)
            cost = c_fix if self.state["EOL"] > self.risk_threshold else c_issue
            reward -= cost
        elif action == 2:
            self.cost_history.append(c_refurbish)
            cost = c_refurbish if self.state["EOL"] > self.risk_threshold else c_issue
            reward -= cost

        if self.state["EOL"] > self.failure_threshold:
            # 50 / conservativism
            reward -= self.conservatism*(self.state["EOL"]*(sum(self.reward_history)))

        return reward
    def scale_state(self):
        scaled_age = (self.state["age"] - self.age_min) / (self.age_max - self.age_min)

        self.state_vector = np.concatenate(
            (self.voltage_one_hot,
             [scaled_age, self.state["failure_rate"], self.state["EOL"]]))

        return self.state_vector

    def scale_cost(self, cost):
        return (cost - self.c_min) / (self.c_max - self.c_min)

    def reset(self):
        #self.substation = random.sample(grid["substations"], k=1)[0]
        self.voltage_one_hot[list(self.voltage_types).index(self.voltage)] = 1

        self.state["age"] = self.initial_age
        self.state["failure_rate"] = self.failure_rate
        self.state["EOL"] = self.eol(self.initial_age)
        self.state["voltage"] = self.voltage_one_hot

        self.state_vector = self.scale_state()

        reward = 0
        done = False

        self.action_history = []
        self.action_costs = []
        self.cost_history = []
        self.reward_history = []
        self.risk = []

        self.reward_history.append(reward)

        return self.state_vector, reward, done, {"time_step": self.time_step}

    def step(self, action):
        """
        Increment Age of Asset in as a Dynamic Process
        Consider Backup Diagram: Compute reward r_t, given a_t
        Compute environment transition from s_t, s_t+1 returning updated state
        """

        self.time_step += 1
        self.degrade(action)
        self.action_history.append(action)

        failed = self.state["EOL"] > self.failure_threshold
        reward = self.compute_reward(action)

        done = failed

        self.state_vector = self.scale_state()
        self.risk.append(self.state["EOL"])
        self.reward_history.append(reward)

        return self.state_vector, reward, done, {"time_step": self.time_step}

    def render(self, run):
        file_name = f"animations/{run}.mp4"
        max_costs = max(np.cumsum(self.cost_history))

        # Specify Fig Size on Launch
        fig, ax1 = plt.subplots(figsize=(15, 10))
        ax2 = ax1.twinx()

        # Create a text box
        action_text = ax1.text(0.02, self.failure_threshold - 0.03, '', transform=ax1.transAxes)
        cost_text = ax1.text(0.02, self.failure_threshold - 0.06, '', transform=ax1.transAxes)
        cumulative_cost_text = ax1.text(0.02, self.failure_threshold - 0.09, '', transform=ax1.transAxes)
        reward_text = ax1.text(0.02, self.failure_threshold - 0.12, '', transform=ax1.transAxes)

        def update(frame):
            ax1.clear()
            ax2.clear()

            ages = np.arange(frame + 1)

            # Reapply the x-axis limits
            ax1.set_ylim(0, 1)
            ax2.set_ylim(0, max_costs)
            ax1.set_xlim(0, self.max_steps)
            ax2.set_xlim(0, self.max_steps)

            # Plot the cumulative change in risk (probability of failure)
            ax1.plot(ages, self.risk[:frame + 1], label='EOL', color='g')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('EOL')
            ax1.legend(loc='upper left')

            blue_1 = (0.0, 1.0, 0.0)

            # Calculate and plot the cumulative change in cost based on the actions
            cumulative_costs = np.cumsum(self.cost_history)
            cost_line, = ax2.plot(ages, cumulative_costs[:frame + 1], color="b")
            cost_line.set_label('Cumulative Cost')
            ax2.legend(loc='upper right')

            # rewards, = ax2.plot(ages, self.reward_history[:frame + 1], color=blue_2)
            # rewards.set_label("Rewards")
            # ax2.legend(loc='upper right')

            red_1 = (1.0, 0.0, 0.0)
            red_2 = (1.0, 0.8, 0.8)

            # Draw the fixed EOL replace cost line
            ax1.axhline(y=self.risk_threshold, color=red_2, linestyle='--', label='high-risk threshold')
            ax1.legend(loc='upper left')

            ax1.axhline(y=self.failure_threshold, color=red_1, linestyle='--', label='failure threshold')
            ax1.legend(loc='upper center')

            padding = -0.10
            # Add annotations for the action taken by the agent
            for i, action in enumerate(self.action_history[:frame + 1]):
                y_coord = self.risk[i] * (1 + padding)
                if (action == 1 or action == 2):
                    ax1.annotate(f'{action}', (i, y_coord), fontsize=15)

            action_text.set_text(f'Action: {self.action_history[frame]}')
            cost_text.set_text(f'Cost: {self.cost_history[frame]}')
            cumulative_cost_text.set_text(f'Cumulative Cost: {round(sum(self.cost_history[:frame + 1]), 2)}')
            reward_text.set_text(f"Reward: {round(sum(self.reward_history[:frame + 1]), 2)}")

            plt.title(
                f'{string.capwords(run)}: Cumulative Risk & Cost \n\n Substation: {self.name}',
                fontweight="bold"
            )

            # Add the text back to the axes after clearing
            ax1.add_artist(action_text)
            ax1.add_artist(cost_text)
            ax1.add_artist(cumulative_cost_text)
            ax1.add_artist(reward_text)

        #ax2.yaxis.set_major_formatter(formatter)

        # Set the x-axis limits for the time horizon
        ax1.set_ylim(0, 1)
        ax2.set_ylim(0, max_costs)
        ax1.set_xlim(0, self.max_steps)
        ax2.set_xlim(0, self.max_steps)

        # Iterate over the time steps and call the update function for each step
        for frame in range(len(self.action_history)):
            update(frame)
            plt.pause(0.1)

        ani = FuncAnimation(fig, update, frames=len(self.action_history), repeat=False)

        # Save the animation. Requires ffmpeg to be installed.
        ani.save(file_name, writer='ffmpeg')
        plt.show()
        plt.close()

# TODO: Parse Args
test_run = False
if test_run:
    env = TransmissionAsset(visualise=True)
    env.reset()
    done = False

    while ((not done) and (env.time_step < env.max_steps)):
        action = env.action_spaces.sample()
        obs, reward, done, rw_state = env.step(action=0)
        print(float(reward))

# env.render(run=f"Substation: {env.name}, Do Nothing")
#env.render(run=f"Substation: {env.name}, Test Run")

    # def compute_reward(self, action, failed):
    #     reward = self.CONSTANT_POSITIVE_REWARD
    #
    #     if action == 0:
    #         #reward -= self.INACTION_PENALTY
    #         self.cost_history.append(self.c_nothing)
    #         #reward -= self.scale_cost(self.c_nothing)
    #         #self.cost_history.append(self.c_nothing)
    #
    #     if action == 1:  # FIX
    #         reward -= self.scale_cost(self.c_fix)
    #         self.cost_history.append(self.c_fix)
    #
    #     elif action == 2:  # REPLACE
    #         reward -= self.scale_cost(self.c_refurbish)
    #         self.cost_history.append(self.c_refurbish)
    #
    #     if failed:
    #         print("UNMANAGED FAILURE")
    #         self.unmanaged_failure = True
    #         reward += self.scale_cost(self.penalty)
    #         self.cost_history.append(self.penalty)
    #
    #     #reward = np.clip(reward, -1, 1)
    #     print(reward)
    #     return reward

# failure event only occurs if EOL > 0.1

# alpha = max(self.state["EOL"] * scale_factor, 1e-3)
# beta = max((1 - self.state["EOL"]) * scale_factor, 1e-3)
# failure_event = np.random.beta(alpha, beta)

# failed = (failure_event > self.state["EOL"]) and (self.state["EOL"] > failure_threshold)