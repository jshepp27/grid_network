import gym
from gym import spaces
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import clear_output

failure_rates = {
    "A": 0.019,
    "B": 0.01,
    "C": 0.01
}
class TransmissionAsset(gym.Env):
    def __init__(self,
                 state_min=0,
                 state_max=1,
                 reward_min=-1,
                 reward_max=1,
                 family_type="A",
                 failure_rates=failure_rates,
                 max_steps=100,
                 visualise=False):

        super(TransmissionAsset, self).__init__()

        self.visualise = visualise

        if self.visualise:
            self.fig, self.ax1 = plt.subplots()
            self.ax2 = self.ax1.twinx()
            self.ani = None

        self.initial_age = 1
        self.unmanaged_failure = False

        self.family_type = family_type
        self.family_type_one_hot = np.zeros(len(failure_rates))

        self.failure_rate = failure_rates[self.family_type]

        self.CONSTANT_POSITIVE_REWARD = 1.0
        self.C_fix = 0.5
        self.C_replace = 0.8
        self.max_steps = max_steps

        # STATE-SCALING
        self.state_min = state_min
        self.state_max = state_max
        self.reward_min = reward_min
        self.reward_max = reward_max

        # ACTION SPACE
        self.action_spaces = spaces.Discrete(3)

        # OBSERVATION SPACE
        self.observation_space = spaces.Dict({
            "age": spaces.Box(low=0, high=float("inf"), shape=(1,)),
            "family_type": spaces.Discrete(len(failure_rates)),
            "failure_rate": spaces.Box(low=0, high=1, shape=(1,)),
            "probability_failure": spaces.Box(low=0, high=1, shape=(1,)),
        })

        self.state = {}
        # TODO: (?) Why Re-shape? Understand Re-shape
        self.state_vector = np.zeros(len(self.observation_space.keys())).reshape(-1, 1)
        self.action_history = []

        self.agent_risk = []
        self.agent_cumulative_cost = []

    def probability_failure(self, age):
        return float(1 - math.exp(-self.failure_rate * age))
    def degrade(self, action):
        # CASE: a_t
        if action == 0:
            # DO NOTHING
            self.state["age"] += 1

        elif action == 1:
            # FIX
            # TODO: Parameterise
            age_reduction = self.state["age"] - np.random.uniform(10, 30)
            self.state["age"] = round(max(self.state["age"] - age_reduction, 1))

        else:
            # REPLACE
            self.state["age"] = 1

        self.state["probability_failure"] = 1 - math.exp(-self.failure_rate * self.state["age"])

    def compute_reward(self, action):
        reward = self.CONSTANT_POSITIVE_REWARD

        if action == 1:  # FIX
            reward -= self.C_fix
        elif action == 2:  # REPLACE
            reward -= self.C_replace

        return reward

    def reset(self):
        self.state["age"] = self.initial_age
        self.family_type_one_hot[list(failure_rates.keys()).index(self.family_type)] = 1
        self.state["failure_rate"] = self.failure_rate
        self.state["probability_failure"] = self.probability_failure(self.initial_age)
        self.state["family_type"] = self.family_type_one_hot

        self.state_vector = np.array([self.state["age"],
                                      self.state["failure_rate"],
                                      self.state["probability_failure"]])

        self.state_vector = np.concatenate(
            (self.family_type_one_hot, [self.state["age"], self.state["failure_rate"], self.state["probability_failure"]]))

        reward = 0
        done = False

        # MIN-MAX SCALING
        self.state_vector = (self.state_vector - self.state_min) / (self.state_max - self.state_min)

        return self.state_vector, reward, done, {}

    def step(self, action):
        """
        Increment Age of Asset in as a Dynamic Process
        Consider Backup Diagram: Compute reward r_t, given a_t
        Compute environment transition from s_t, s_t+1 returning updated state
        """

        # Increment Age by timestep
        self.degrade(action)
        self.action_history.append(action)

        reward = self.compute_reward(action)

        done = False

        if self.state["probability_failure"] >= 0.91:
            print("UNMANAGED FAILURE")
            self.unmanaged_failure = True
            done = True
            penalty = reward - 200
            reward += (penalty - self.reward_min) / (self.reward_max - self.reward_min)

        self.state_vector = np.array([self.state["age"],
                                      self.state["failure_rate"],
                                      self.state["probability_failure"]])

        self.state_vector = np.concatenate(
            (self.family_type_one_hot, [self.state["age"], self.state["failure_rate"], self.state["probability_failure"]]))

        # MIN-MAX SCALING
        self.state_vector = (self.state_vector - self.state_min) / (self.state_max - self.state_min)

        # Reward scaling
        reward = (reward - self.reward_min) / (self.reward_max - self.reward_min)

        self.agent_risk.append(self.state["probability_failure"])
        self.agent_cumulative_cost.append(sum(self.action_history))

        return self.state_vector, reward, done, self.state

    def render(self):
        eol_replace_cost = 91
        def update(frame):
            ax1.clear()
            ax2.clear()

            ages = np.arange(frame + 1)

            # Reapply the x-axis limits
            ax1.set_ylim(0, 1)
            ax2.set_ylim(0, 5)
            ax1.set_xlim(0, self.max_steps)
            ax2.set_xlim(0, self.max_steps)

            # Plot the cumulative change in risk (probability of failure)
            ax1.plot(ages, self.agent_risk[:frame + 1], label='Cumulative Risk (Probability of Failure)', color='r')
            ax1.set_xlabel('Ages')
            ax1.set_ylabel('Risk (Probability of Failure)')
            ax1.legend(loc='upper left')

            # Calculate and plot the cumulative change in cost based on the actions
            cost = [0 if action == 0 else self.C_fix if action == 1 else self.C_replace for action in self.action_history[:frame + 1]]
            cumulative_cost = np.cumsum(cost)
            cost_line, = ax2.plot(ages, cumulative_cost, color='b')
            cost_line.set_label('Cumulative Cost')
            ax2.legend(loc='upper right')

            # Draw the fixed EOL replace cost line
            ax2.axhline(y=eol_replace_cost, color='g', linestyle='--', label='EOL Replace Cost')
            ax2.legend(loc='upper right')

            # padding = 0.5
            # # Add annotations for the action taken by the agent
            # for i, action in enumerate(self.action_history[:frame + 1]):
            #     y_coord = self.agent_risk[i] * (1 + padding)
            #     ax1.annotate(f'{action}', (i, y_coord), fontsize=10)

            plt.title('Cumulative Risk and Cost Performance')

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        # Set the x-axis limits for the time horizon
        ax1.set_ylim(0, 1)
        ax2.set_ylim(0, 5)
        ax1.set_xlim(0, self.max_steps)
        ax2.set_xlim(0, self.max_steps)

        # Iterate over the time steps and call the update function for each step
        for frame in range(len(self.action_history)):
            update(frame)
            plt.pause(0.1)

# env = TransmissionAsset(visualise=True)
# env.reset()
# for _ in range(100):
#     action = env.action_spaces.sample()
#     state, reward, done, _ = env.step(action=action)
#     print(done)
#
# env.render()
# plt.show()


"""
TODOs:
- Perform Data Analysis and Convert to Real-Units 

- Introduce age vs enhanced_age, for better visualisation 
- Consider ISO Physical Asset Management Framework

- Render Visualisation that depicts change in risk as Agent takes actions [DONE]
- Implement State as an Update to Dynamic Processes [DONE]
- Implement Reward [DONE]
- Scale State and Reward as Normalised Values [DONE]

Consider:
- Rendering 
- Animated Plots
- Pillow
- Evaluating on EOL extension, age asset maintained on the Network, Cost-Benefit
"""

### WORKS ###
# def render(self):
#     def update(frame):
#         ax.clear()
#         ax.plot(np.arange(frame + 1), np.arange(frame + 1), color='r')
#         ax.set_xlim(0, self.max_steps)
#         ax.set_ylim(0, self.max_steps)
#         plt.title('Linear Line')
#         return plt
#
#     fig, ax = plt.subplots()
#
#     for frame in range(self.max_steps):
#         update(frame)
#         plt.pause(0.1)

### WORKING ###
# def render(self):
#     def update(frame):
#         ax1.clear()
#         ax2.clear()
#
#         ages = list(range(len(self.action_history)))
#
#         # Plot the risk baseline (degradation curve) and agent risk
#         ax1.plot(range(self.max_steps), risk_baseline, label='Risk Baseline (Degradation Curve)', linestyle='--')
#         ax1.plot(ages, self.agent_risk, label='Agent Risk', color='r')
#         ax1.set_xlabel('Ages')
#         ax1.set_ylabel('Risk (Probability of Failure)')
#         ax1.legend(loc='upper left')
#
#         # Plot the cost baseline (total replacement cost) and agent cumulative cost
#         ax2.axhline(cost_baseline, label='Cost Baseline (Total Replacement Cost)', linestyle='--', color='g')
#         ax2.plot(ages, self.agent_cumulative_cost, label='Agent Cumulative Cost', color='b')
#         ax2.set_ylabel('Cost')
#         ax2.legend(loc='upper right')
#
#         plt.title('Agent Risk and Cost Performance')
#
#     fig, ax1 = plt.subplots()
#     ax2 = ax1.twinx()
#
#     # Generate the risk baseline (degradation curve)
#     risk_baseline = [self.probability_failure(age) for age in range(self.max_steps)]
#
#     # Calculate the cost baseline (total replacement cost)
#     cost_baseline = self.C_replace
#
#     ani = FuncAnimation(fig, update, frames=len(self.action_history), repeat=False)
#     plt.pause(0.1)

### SIMPLE NO ANIMATION ###

# def render(self):
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
#
#     # Generate the risk baseline (degradation curve)
#     risk_baseline = [self.probability_failure(age) for age in range(self.max_steps)]
#
#     # Calculate the cost baseline (total replacement cost)
#     cost_baseline = self.C_replace
#
#     # Create line objects for each plot
#     risk_line, = ax1.plot([], [], label='Agent Risk', color='r')
#     risk_baseline_line, = ax1.plot([], [], label='Risk Baseline (Degradation Curve)', linestyle='--')
#     cost_line, = ax2.plot([], [], label='Agent Cumulative Cost', color='b')
#     cost_baseline_line, = ax2.plot([], [], label='Cost Baseline (Total Replacement Cost)', linestyle='--',
#                                    color='g')
#
#     plt.suptitle('Agent Risk and Cost Performance', fontsize=16)
#     plt.tight_layout()
#
#     # Store the current state of the plot
#     current_data = {'risk': [], 'risk_baseline': [], 'cost': [], 'cost_baseline': []}
#
#     for frame in range(len(self.action_history)):
#         ages = list(range(frame + 1))
#
#         # Update the data of each line object
#         current_data['risk'] = self.agent_risk[:frame + 1]
#         current_data['risk_baseline'] = risk_baseline
#         current_data['cost'] = self.agent_cumulative_cost[:frame + 1]
#         current_data['cost_baseline'] = [cost_baseline, cost_baseline]
#
#         risk_line.set_data(ages, current_data['risk'])
#         risk_baseline_line.set_data(range(self.max_steps), current_data['risk_baseline'])
#         cost_line.set_data(ages, current_data['cost'])
#         cost_baseline_line.set_data([0, len(self.action_history)], current_data['cost_baseline'])
#
#         ax1.relim()
#         ax1.autoscale_view()
#         ax2.relim()
#         ax2.autoscale_view()
#
#         plt.draw()
#         plt.pause(0.1)

   # def render(self):
    #     T = len(self.action_history)
    #     ages = list(range(T))
    #
    #     # Generate the risk baseline (degradation curve)
    #     risk_baseline = [self.probability_failure(age) for age in ages]
    #
    #     # Generate the agent risk data
    #     agent_risk = [self.probability_failure(self.state["age"]) for self.state["age"] in ages]
    #
    #     # Calculate the cost baseline (total replacement cost)
    #     cost_baseline = [self.C_replace] * T
    #
    #     # Generate the agent cumulative cost data
    #     agent_cost = [self.compute_reward(action) for action in self.action_history]
    #     agent_cumulative_cost = np.cumsum(agent_cost)
    #
    #     # Plotting
    #     fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    #
    #     # Plot the risk baseline (degradation curve) and agent risk
    #     ax1.plot(ages, risk_baseline, label='Risk Baseline (Degradation Curve)', linestyle='--')
    #     ax1.plot(ages, agent_risk, label='Agent Risk', color='r')
    #     ax1.set_xlabel('Ages')
    #     ax1.set_ylabel('Risk (Probability of Failure)')
    #     ax1.legend(loc='upper left')
    #
    #     # Plot the cost baseline (total replacement cost) and agent cumulative cost
    #     ax2.plot(ages, cost_baseline, label='Cost Baseline (Total Replacement Cost)', linestyle='--', color='g')
    #     ax2.plot(ages, agent_cumulative_cost, label='Agent Cumulative Cost', color='b')
    #     ax2.set_xlabel('Ages')
    #     ax2.set_ylabel('Cost')
    #     ax2.legend(loc='upper right')
    #
    #     plt.suptitle('Agent Risk and Cost Performance')
    #     plt.tight_layout()
    #     plt.pause(0.1)


    # def render(self):
    #     fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    #
    #     # Generate the risk baseline (degradation curve)
    #     risk_baseline = [self.probability_failure(age) for age in range(self.max_steps)]
    #
    #     # Plot the risk baseline (degradation curve) and agent risk
    #     risk_line, = ax1.plot(range(self.max_steps), risk_baseline, label='Risk Baseline (Degradation Curve)',
    #                           linestyle='--')
    #     agent_risk_line, = ax1.plot([], [], label='Agent Risk', color='r')
    #     ax1.set_xlabel('Ages')
    #     ax1.set_ylabel('Risk (Probability of Failure)')
    #     ax1.legend(loc='upper left')
    #
    #     # Calculate the cost baseline (total replacement cost)
    #     cost_baseline = self.C_replace
    #
    #     # Plot the cost baseline (total replacement cost) and agent cumulative cost
    #     cost_line, = ax2.plot([], [], label='Cost Baseline (Total Replacement Cost)', linestyle='--', color='g')
    #     agent_cost_line, = ax2.plot([], [], label='Agent Cumulative Cost', color='b')
    #     ax2.set_ylabel('Cost')
    #     ax2.legend(loc='upper right')
    #
    #     plt.title('Agent Risk and Cost Performance')
    #
    #     def update(frame):
    #         age = len(self.action_history)
    #         agent_risk = self.state["probability_failure"]
    #         agent_cumulative_cost = sum(self.action_history)
    #
    #         agent_risk_line.set_data(range(age), self.agent_risk)
    #         agent_cost_line.set_data(range(age), self.agent_cumulative_cost)
    #         cost_line.set_data([0, age], [cost_baseline, cost_baseline])
    #
    #     # Set the interval and number of frames to match the number of steps in the episode
    #     interval = 50
    #     frames = range(0, len(self.action_history), interval)
    #
    #     ani = FuncAnimation(fig, update, frames=frames, repeat=False)
    #     plt.pause(0.1)


#plt.show()
# def plot_probability_of_failure(env):
#     age = env.state["age"]
#     failure_rate = env.state["failure_rate"]
#     probability_failure = env.state["probability_failure"]
#
#     fig, ax = plt.subplots()
#     ax.plot(range(age + 1), [env.probability_failure(i) for i in range(age + 1)])
#     ax.axhline(y=0.91, color='r', linestyle='--')
#
#     ax.set_xlabel('Age')
#     ax.set_ylabel('Probability of Failure')
#     ax.set_ylim(0, 1)
#
#     plt.show(block=False)  # Add block=False to prevent window from blocking
#     plt.pause(0.5)  # Add pause to wait before updating
#     plt.close(fig)  # Add close to clear the figure for the next frame

# def compute_reward(self, action):
#     reward = self.probability_failure(age=self.state["age"])
#
#     if action == 1:
#         #reward -= self.C_fix if self.state["probability_failure"] > 0.5 else 1
#         reward_adjustment = self.state["probability_failure"] if self.state["probability_failure"] > 0.5 else 1
#         reward -= self.C_fix * reward_adjustment
#     elif action == 2:
#         #reward -= self.C_replace if self.state["probability_failure"] > 0.5 else 1
#         reward_adjustment = self.state["probability_failure"] if self.state["probability_failure"] > 0.5 else 1
#         reward -= self.C_replace * reward_adjustment
#
#     return reward