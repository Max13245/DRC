import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

cuda_available = torch.cuda.is_available()

# Use gpu if present on current device
device = torch.device("cuda" if cuda_available else "cpu")

# Number of episodes is the same as epochs TODO: Change variable if needed (for overfitting) (Depends on testroom)
if cuda_available:
    N_EPISODES = 600
else:
    N_EPISODES = 50

# TODO: Decide actual number (is equal to audio fragment)
N_STEPS = 1000

# TODO: next_state might not be needed here
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, state, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(state, 128)  # Input layer
        self.layer2 = nn.Linear(128, 128)  # Hidden layer
        self.layer3 = nn.Linear(128, n_actions)  # Output layer

    def forward(self, x):
        x = F.relu(self.layer1(x))  # TODO: Not sure about relu
        x = F.relu(self.layer2(x))  # TODO: Not sure about relu
        x = F.sigmoid(self.layer3(x))
        return x


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# TODO: Make a program to get states and perform actions
# Get number of actions that the network can take (output is five numbers from 0 - 1)
N_ACTIONS = 5
# Get the number of state observations (inputs for the network) TODO: don't set state to None
state, info = None
N_OBSERVATIONS = len(state)

policy_net = DQN(N_OBSERVATIONS, N_ACTIONS).to(device)
target_net = DQN(N_OBSERVATIONS, N_ACTIONS).to(device)
target_net.load_state_dict(policy_net.state_dict())

# TODO: Why this optimizer?
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


def select_action(state, step: int):
    # Use eps_threshold to have a good balance between exploration and exploitation
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * step / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # TODO: This chooses one action, not needed in this program,
            # since there is an infinite number of actions (Makes it hard
            # to choose :))
            # Don't choose (maybe change the randomness aswell)
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # TODO: Choose a random action
        return torch.tensor([[...]], device=device, dtype=torch.long)


def optimize_model() -> None:
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute Huber loss TODO: Is Huber loss the way to goooooo?
    criterion = nn.SmoothL1Loss()
    # Warning: Not sure about unsqueeze when only using rewarch_batch
    loss = criterion(state_action_values, reward_batch.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping TODO: Look if needed and what good for
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def train_loop():
    # TODO: transform some inputs to tensors
    for episode in range(N_EPISODES):
        # Reset the state (pause, so there are no echoes anymore)
        # Get state
        for step in N_STEPS:
            # Take step and collect observation room (new state), reward,
            # termination (not applicable) and truncated.

            total_steps = episode * step
            action = select_action(state, total_steps)
            new_state, reward, truncated = take_step(action)  # One step is...
