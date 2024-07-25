import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define the DQN model class
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# Load the trained model
env = gym.make("CartPole-v1", render_mode="rgb_array")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state, info = env.reset()
n_observations = len(state)
n_actions = env.action_space.n

policy_net = DQN(n_observations=n_observations, n_actions=n_actions).to(device)
policy_net.load_state_dict(torch.load("laboratory_3/policy_net.pth"))

def select_action(state):
    with torch.no_grad():
        return policy_net(state).max(1).indices.view(1, 1)

# Function to run the episode and capture frames
def run_episode(env, policy_net):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    frames = []
    done = False
    while not done:
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        frames.append(env.render())
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    return frames

# Run a few episodes and capture frames
episodes = 3
frames = []
for _ in range(episodes):
    frames.extend(run_episode(env, policy_net))

# Create an animation
fig = plt.figure()
patch = plt.imshow(frames[0])
plt.axis('off')

def animate(i):
    patch.set_data(frames[i])

anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50)
plt.show()

env.close()
