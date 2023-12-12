import math
import random
from collections import namedtuple, deque
import csv
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from room_simulator.room_sim import AcousticRoom


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


# Use gpu if present on current device
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

# TODO: next_state might not be needed here
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
LR = 1e-4

# TODO: Make a program to get states and perform actions
# Get number of actions that the network can take (output is five numbers from 0 - 1)
N_OUPUTS = 5
# Get the number of state observations (inputs for the network) TODO: don't set state to None
state, info = None, None
N_OBSERVATIONS = 35  # len(state)

policy_net = DQN(N_OBSERVATIONS, N_OUPUTS).to(device)
target_net = DQN(N_OBSERVATIONS, N_OUPUTS).to(device)
target_net.load_state_dict(policy_net.state_dict())

# TODO: Why this optimizer?
network = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)


def select_action(state, step: int):
    # Use eps_threshold to have a good balance between exploration and exploitation
    random_balance = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * step / EPS_DECAY)
    if random_balance > eps_threshold:
        with torch.no_grad():
            return policy_net(state)
    else:
        # Random action (five random numbers between 0 and 1)
        return [random.uniform(0, 1) for _ in range(0, 5)]


def optimize_model(current_state, target_state) -> None:
    # Use L1Loss, because it suports complex numbers
    # TODO: Doesn't punish higher loss as much as other loss functions
    criterion = nn.L1Loss()

    current_state.requires_grad_()
    target_state.requires_grad_()

    # Warning: Not sure about unsqueeze when only using rewarch_batch
    loss = criterion(current_state, target_state)
    print(f"Loss: {loss}")

    # Optimize the model
    network.zero_grad()
    loss.backward()
    # In-place gradient clipping TODO: Look if needed and what good for
    # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    network.step()


def get_train_data(file_path: str) -> list:
    with open(file_path, mode="r") as file:
        csv_file = csv.reader(file)
        for row in csv_file:
            yield row


def format_static_state(room: AcousticRoom) -> list:
    """
    Inputs of the Neural Network are:
    1. Frequency, Dynamic
    2. Amplitude, Dynamic
    3. Location speakers (Optional, but way better), Static
    4. Location mic (Optional, but way better), Static
    5. Room measurements (Optional, a bit better, since relfection also takes care of this a bit), Static
    6. Reflection level (For each speaker?) with sweeptonetest, Static
    """
    speaker_positions = [dim for position in room.speaker_positions for dim in position]
    return speaker_positions + list(room.mic_position) + list(room.reflection_levels)


def train_loop():
    """
    One episode will be a sound played in a unique room
    There are no individual steps, since it is not possible
    to stop in the middle of a sound. For training during the
    room simulation each audio fragment (one step) will be inputted
    into the NN this audio will be adjusted for each speaker.
    Before the audio is put trough the NN it will be adjusted
    to the master volume.
    """

    # Get episode data from train data file
    data = get_train_data("./neural_network/train_data_input.csv")

    # Number of episodes is determinded from test data input file
    for n_episode, episode in enumerate(data):
        room = AcousticRoom(episode)

        # Get inputs that don't change during current episode
        static_input = format_static_state(room)

        # Sample the audio
        fft_samples = room.get_fft_audio(room.master_audio)

        speaker_audios = [[] for _ in range(5)]
        for sample_indx, sample in enumerate(fft_samples):
            # Get the peaks of the sepparated waves
            indices = room.get_significant_waves(sample[1])

            # Multiply sample in 5 different streams for five different speakers
            # Make a copy of array, otherwise connected to each other (same pointer)
            for indx in range(0, len(speaker_audios)):
                speaker_audios[indx].append(np.copy(sample[1]))

            for peak_index in indices[0 : int(len(indices) / 2)]:
                # Create a state for the neural network
                # Don't normalize fft input, original fft needed for ifft
                frequency = sample[0][peak_index]
                amplitude = np.abs(sample[1][peak_index])
                formatted_input = static_input + [frequency, amplitude]
                tensor_input = torch.FloatTensor(formatted_input).requires_grad_()

                # Get scalers for the peaks for each speaker
                scalers = select_action(tensor_input, n_episode)

                # Loop over speakers, to multiply peaks with scalers
                for speaker_indx in range(0, len(speaker_audios)):
                    speaker_scaler = scalers[speaker_indx]

                    # Scale both the amplitudes for negative and positive frequency peaks
                    speaker_audios[speaker_indx][sample_indx][
                        peak_index
                    ] *= speaker_scaler
                    speaker_audios[speaker_indx][sample_indx][
                        -peak_index
                    ] *= speaker_scaler

        # Create 5 streams for the speakers
        streams = [room.get_ifft_audio(stream) for stream in speaker_audios]

        # Add speakers to the room
        room.add_speakers(streams)

        # Add microphone to the room
        room.room.add_microphone(room.mic_position)

        # Simulate the room
        room.room.simulate(recompute_rir=True)

        # Get recorded sound
        """room.room.mic_array.to_wav(
            "./neural_network/first_test.wav", norm=True, bitdepth=np.int16
        )"""

        # Store recorded sound
        room.store_recorded_audio()

        # Get fft samples from recorded audio
        recorded_fft_samples = room.get_fft_audio(room.recorded_audio)

        recorded_amplitudes = np.array(
            [np.array(fft_sample[1]) for fft_sample in recorded_fft_samples]
        )
        master_amplitudes = np.array(
            [np.array(fft_sample[1]) for fft_sample in fft_samples]
        )

        recorded_amplitudes = torch.tensor(recorded_amplitudes, dtype=torch.complex128)
        master_amplitudes = torch.tensor(master_amplitudes, dtype=torch.complex128)

        optimize_model(recorded_amplitudes, master_amplitudes)


def config_loop():
    """
    Use when training/configuring in a real room
    Incase of a real room, no plan like with room
    simulation
    """


train_loop()
