"""Network and replay buffer for the DQN agent.

Kept separate from callbacks.py so the architecture can be unit-tested and
swapped without touching the game interface.
"""

import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn

# One entry per channel produced by state_to_features. Channels 1, 3 and 5 are
# all-zero in the coin-heaven scenario but are present from the start so the
# same architecture carries over to tasks 2-4 without retraining from scratch.
CHANNELS = ("walls", "crates", "self", "others", "coins", "danger")
N_CHANNELS = len(CHANNELS)

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
N_ACTIONS = len(ACTIONS)

# Curriculum stages may restrict the action space. Task 1 (coin-heaven) never
# requires a bomb, and an exploring agent with BOMB available spends its first
# thousand rounds killing itself instead of learning to walk. Restricting the
# set is a documented stage choice, not a shortcut: the network still has to
# learn which of the remaining actions is best, and the mask is removed for the
# later tasks. The head always has all six outputs so weights transfer.
MOVES_ONLY = ["UP", "RIGHT", "DOWN", "LEFT"]


def action_mask(allowed) -> torch.Tensor:
    """Boolean tensor over ACTIONS, True where the action may be selected."""
    mask = torch.zeros(N_ACTIONS, dtype=torch.bool)
    for name in allowed:
        mask[ACTIONS.index(name)] = True
    return mask


def masked_argmax(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """argmax over the last dimension, ignoring disallowed actions."""
    return q_values.masked_fill(~mask, float("-inf")).argmax(dim=-1)

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))


class QNetwork(nn.Module):
    """Small convolutional Q-network.

    Input  : (batch, N_CHANNELS, rows, cols)
    Output : (batch, N_ACTIONS) estimated Q values.

    Deliberately small. It has to run a forward pass inside the 0.5 s per-step
    budget on one thread of a Ryzen 5 2600, and a big network mostly buys us a
    longer wait for convergence rather than a better coin collector.
    """

    def __init__(self, rows: int, cols: int, n_channels: int = N_CHANNELS,
                 n_actions: int = N_ACTIONS, width: int = 32):
        super().__init__()
        self.rows = rows
        self.cols = cols

        self.features = nn.Sequential(
            nn.Conv2d(n_channels, width // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width // 2, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width // 2, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        flat = (width // 2) * rows * cols
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class ReplayBuffer:
    """Uniform experience replay.

    Prioritised replay is a planned upgrade. Uniform first, so that the
    prioritised version has an honest baseline to be compared against in the
    report rather than being adopted on faith.
    """

    def __init__(self, capacity: int, rng: random.Random = None):
        self.memory = deque(maxlen=capacity)
        self.rng = rng or random.Random()

    def push(self, state, action, next_state, reward, done):
        self.memory.append(Transition(state, action, next_state, reward, done))

    def sample(self, batch_size: int):
        batch = self.rng.sample(self.memory, batch_size)

        states = torch.from_numpy(np.stack([t.state for t in batch])).float()
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32)

        # Terminal transitions have no successor. Feed a zero state and mask it
        # out with `dones` so the shapes stay rectangular.
        zero = np.zeros_like(batch[0].state)
        next_states = torch.from_numpy(
            np.stack([t.next_state if t.next_state is not None else zero for t in batch])
        ).float()

        return states, actions, next_states, rewards, dones

    def __len__(self):
        return len(self.memory)
