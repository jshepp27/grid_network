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
"""

failure_rates = {
    "A": 0.0045,
    "B": 0.0025,
    "C": 0.0035
}
class TransmissionAsset(gym.Env):
    def __init__(self):
        super(TransmissionAsset, self).__init__()

        self.initial_age = 1
        self.unmanaged_failure = False

        self.family_type = "A"
        self.family_type_one_hot = np.zeros(len(failure_rates))

        self.failure_rate = failure_rates[self.family_type]

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
        self.state_vector = np.zeros(len(self.observation_space.keys())).reshape(-1, 1)
    def probability_failure(self, age):
        return float(1 - math.exp(-self.failure_rate * age))
    def degrade(self, action=0):
        # CASE: a_t
        if action == 0:
            # DO NOTHING
            self.state["age"] += 1

        elif action == 1:
            # FIX
            age_reduction = self.state["age"] - np.random.uniform(10, 30)
            self.state["age"] = round(max(self.state["age"] - age_reduction, 1))

        else:
            # REPLACE
            self.state["age"] = 1

        self.state["probability_failure"] = 1 - math.exp(-self.failure_rate * self.state["age"])

    def compute_reward(self, action):
        reward = self.probability_failure(age=self.state["age"])

        if action == 1:
            reward -= 0.5 if self.state["probability_failure"] > 0.5 else 1
        elif action == 2:
            reward -= 0.8 if self.state["probability_failure"] > 0.5 else 1

        return reward

    def reset(self):
        self.state["age"] = self.initial_age
        self.family_type_one_hot[list(failure_rates.keys()).index(self.family_type)] = 1
        self.state["failure_rate"] = self.failure_rate
        self.state["probability_failure"] = self.probability_failure(self.initial_age)

        self.state_vector = np.array([self.state["age"],
                                      self.state["failure_rate"],
                                      self.state["probability_failure"]])

        self.state_vector = np.concatenate(
            (self.family_type_one_hot, [self.state["age"], self.state["failure_rate"], self.state["probability_failure"]]))

        reward = 0
        done = False

        return self.state_vector, reward, done, {}

    def step(self, action):
        """
        Increment Age of Asset in as a Dynamic Process
        Compute reward r_t, given a_t
        Compute environment transition from s_t, s_t+1 returning updated state
        """

        # Increment Age by timestep
        self.degrade(action)

        reward = self.compute_reward(action)

        done = False
        if self.state["probability_failure"] >= 0.95:
            #print("FAILED AT:", self.state["probability_failure"])
            print("UNMANAGED FAILURE")
            self.unmanaged_failure = True
            done = True
            reward -= 100

        # elif np.random.randint(0,1) > 0.95:
        #     print("RANDOM FAILURE")
        #     done = True

        self.state_vector = np.array([self.state["age"],
                                      self.state["failure_rate"],
                                      self.state["probability_failure"]])

        self.state_vector = np.concatenate(
            (self.family_type_one_hot, [self.state["age"], self.state["failure_rate"], self.state["probability_failure"]]))

        return self.state_vector, reward, done, self.state
    def render(self):
        pass

# env = TransmissionAsset()
#
# env.reset()
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

