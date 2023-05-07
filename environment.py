import gym
from gym import spaces
import math
import numpy as np

import inspect
from pprint import pprint

"""
TODOs:
- Implement State as an Update to Dynamic Processes
- Implement Reward
- Implement Render Visualisation
- Scale State and Reward as Normalised Values
- Vary (parameterise) Family Type
- USe Historical Data

Consider:
- Rendering 
- Animated Plots
- Pillow
- Evaluating on EOL extension, age asset maintained on the Network, Cost-Benefit
"""

failure_rates = {
    "A": 0.010,
    "B": 0.008,
    "C": 0.009
}
class TransmissionAsset(gym.Env):
    def __init__(self,
                 state_min=0,
                 state_max=1,
                 reward_min=-1,
                 reward_max=1,
                 family_type="A",
                 failure_rates=failure_rates):

        super(TransmissionAsset, self).__init__()

        self.initial_age = 1
        self.unmanaged_failure = False

        self.family_type = family_type
        self.family_type_one_hot = np.zeros(len(failure_rates))

        self.failure_rate = failure_rates[self.family_type]

        self.C_fix = 0.5
        self.C_replace = 0.8

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
        reward = self.probability_failure(age=self.state["age"])

        if action == 1:
            reward -= self.C_fix if self.state["probability_failure"] > 0.5 else 1
        elif action == 2:
            reward -= self.C_replace if self.state["probability_failure"] > 0.5 else 1

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

        reward = self.compute_reward(action)

        done = False

        if self.state["probability_failure"] >= 0.91:
            print("UNMANAGED FAILURE")
            self.unmanaged_failure = True
            done = True
            penalty = reward - 1000
            reward += (penalty - self.reward_min) / (self.reward_max - self.reward_min)

        # RANDOM FAILURE
        # elif np.random.randint(0,1) > 0.95:
        #     print("RANDOM FAILURE")
        #     done = True

        self.state_vector = np.array([self.state["age"],
                                      self.state["failure_rate"],
                                      self.state["probability_failure"]])

        self.state_vector = np.concatenate(
            (self.family_type_one_hot, [self.state["age"], self.state["failure_rate"], self.state["probability_failure"]]))

        # MIN-MAX SCALING
        self.state_vector = (self.state_vector - self.state_min) / (self.state_max - self.state_min)

        # Reward scaling
        reward = (reward - self.reward_min) / (self.reward_max - self.reward_min)

        return self.state_vector, reward, done, self.state
    def render(self):
        # TODO: Render Tabluar Representation
        pass


# env = TransmissionAsset()
#
# env.reset()
# for _ in range(5):
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

