import gym
import numpy as np
from gym import spaces
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib import animation

# TODOs: Visualise Environment State
# TODOs: Visualise Environment Improvement, delta through Work
# TODOs: Visualise Network
# TODOs: Visualise Agent Actions, Network target(s)
# TODOs: Visualise Agent Reward
# TODOs: Visualise as Animation

class TransmissionOperatorEnv(gym.Env):
    def __init__(self):
        super(TransmissionOperatorEnv, self).__init__()

        self.num_assets = 5
        self.family_type_map = {"A": 0, "B": 1, "C": 2}

        # Action space: (asset_index, action_type)
        self.action_space = spaces.MultiDiscrete([self.num_assets, 3])

        # State space: age, family_type, failure_rate, pf for each asset
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.num_assets, 4), dtype=np.float32)

        self.max_time_steps = 100
        self.current_time_step = 0

    def reset(self):
        self.current_time_step = 0

        # Initialize the asset states
        self.state = np.zeros((self.num_assets, 4))
        for i in range(self.num_assets):
            # Random Initilization of the Network
            age = np.random.randint(1, 50)
            family_type = np.random.choice(["A", "B", "C"])
            family_type_encoded = self.family_type_map[family_type]
            failure_rate = np.random.uniform(0.01, 0.2)

            if family_type == "A":
                failure_rate *= 1
            elif family_type == "B":
                failure_rate *= 1.5
            else:  # family_type == "C"
                failure_rate *= 2

            pf = 1 - np.exp(-failure_rate * age)
            self.state[i, :] = np.array([age, family_type_encoded, failure_rate, pf])

        return self.state

    def step(self, action):
        asset_index, action_type = action
        asset_state = self.state[asset_index]

        age, family_type, failure_rate, pf = asset_state

        if action_type == 0:  # do nothing
            age += 1
        elif action_type == 1:  # repair
            age_reduction = np.random.uniform(0.1 * age, 0.4 * age)
            age = max(age - age_reduction, 1)
        else:  # replace
            age = 1

        pf = 1 - np.exp(-failure_rate * age)

        self.state[asset_index, :] = np.array([age, family_type, failure_rate, pf])

        reward = self.calculate_reward(asset_state, action_type)
        self.current_time_step += 1
        done = self.current_time_step >= self.max_time_steps

        return self.state, reward, done, {}

    def calculate_reward(self, asset_state, action):
        age, family_type, failure_rate, pf = asset_state

        if action == 0:  # do nothing
            if pf > 0.9:
                reward = -1
            else:
                reward = 0
        elif action == 1:  # repair
            if pf > 0.5:
                reward = -0.5
            else:
                reward = -1
        else:  # replace
            if pf > 0.5:
                reward = -0.8
            else:
                reward = -1

        network_reward = self.evaluate_network_state()
        return reward + network_reward

    def evaluate_network_state(self):
        total_pf = np.sum(self.state[:, 3])  # Sum of failure probabilities for all assets
        network_size = self.state.shape[0]

        if total_pf / network_size < 0.5:
            return 0.1  # Small bonus for having a healthy network
        elif total_pf / network_size >= 0.5 and total_pf / network_size < network_size >= 0.5 and total_pf / network_size < 0.9:
            return 0  # No reward or penalty for a moderately healthy network
        else:
            return -0.1  # Penalty for having a poorly maintained network

    def render(self, mode='human'):
        headers = ['Asset', 'Age', 'Family Type', 'Failure Rate', 'Failure Probability']
        table_data = []

        for i, asset_state in enumerate(self.state):
            age, family_type, failure_rate, pf = asset_state
            table_data.append([i, age, family_type, failure_rate, pf])

        print(tabulate(table_data, headers=headers))

    def sample_action(self):
        asset_index = np.random.randint(self.num_assets)
        action_type = np.random.randint(3)
        return asset_index, action_type

def create_plot(n_episodes, total_rewards):
    fig, ax = plt.subplots()
    ax.plot(range(1, n_episodes + 1), total_rewards, label="rewards")
    ax.set_title("Total Rewards per Episode")
    return fig, ax


def create_animation(fig, ax, n_episodes, total_rewards, filename, writer='ffmpeg', fps=30):
    def animate(i):
        ax.clear()
        ax.plot(range(1, i + 1), total_rewards[:i], label="rewards")
        ax.set_title("Total Rewards per Episode")
        ax.legend()
        ax.set_xlabel("Time")

    ani = animation.FuncAnimation(fig, animate, frames=n_episodes, repeat=True)
    ani.save(filename, writer=writer, fps=fps)

    plt.show()

    return ani

# Testing the environment with a non-intelligent agent
if __name__ == "__main__":
    # Create the environment
    env = TransmissionOperatorEnv()

    # Number of episodes
    n_episodes = 100

    # Initialise performance stores
    total_rewards = []

    # Loop through episodes
    for episode in range(n_episodes):
        # Reset the environment
        state = env.reset()

        done = False
        total_reward = 0

        # Loop through time steps
        while not done:
            # Select a random action
            action = env.sample_action()

            # Perform the action and receive the new state, reward, and done flag
            next_state, reward, done, _ = env.step(action)

            # Render the environment
            if env.current_time_step % 10 == 0:
                env.render()

            # Update the state and accumulate the reward
            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1}: Total reward = {total_reward}")

        # Append the results to the lists
        total_rewards.append(total_reward)

    # Close the environment
    env.close()

# Plot the total rewards per episode
#plt.plot(range(1, n_episodes+1), total_rewards)
fig, ax = create_plot(n_episodes, total_rewards)

filename = "data/test.mp4"
create_animation(fig=fig, ax=ax, n_episodes=n_episodes, total_rewards=total_rewards, filename=filename)

# fig, ax = plt.subplots()
# ax.plot(range(1, n_episodes+1), total_rewards, label="scores")
# ax.set_title("Total Rewards per Episdoe")
# plt.show()
