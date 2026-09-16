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

# next_legal: which actions are selectable in the SUCCESSOR state. The Double
# DQN target takes an argmax over the next state, and actions the agent can
# never take there carry untrained noise that would win that argmax and
# poison the target. Legality is per-state, so it has to travel with the
# transition rather than being a single fixed mask.
Transition = namedtuple("Transition",
                        ("state", "action", "next_state", "reward", "done", "next_legal"))


# --- 8-fold dihedral augmentation -------------------------------------------
#
# Daniela's suggestion, and the cheapest remaining source of sample efficiency.
# The board is square and every rule of the game is invariant under the eight
# symmetries of a square, so one observed transition is really eight. Rotating
# a board 90 degrees does not change which move is correct, it changes what that
# move is *called*, and the network currently has to learn each orientation from
# separate experience.
#
# Applied at sampling time rather than at push time. Same buffer, no memory
# cost, and each transition is replayed in a different orientation on each
# visit, which also acts as a regulariser. Pushing eight copies instead would
# cut the number of distinct situations the buffer holds by eight.
#
# `state_to_features` builds channels as channels[c][x, y], so tensor dim -2 is
# x and dim -1 is y. Under torch/numpy rot90 in that plane an offset (u, v)
# maps to (-v, u), which sends UP -> RIGHT -> DOWN -> LEFT, i.e. exactly one
# step along ACTIONS[0:4]. Flipping x fixes UP and DOWN and swaps RIGHT and
# LEFT. WAIT and BOMB are invariant under all eight. This is derived, and also
# asserted against the real feature pipeline in tools/test_augment.py, because
# a silently wrong action permutation would teach the network that walking left
# leads where walking up leads, and nothing about the loss curve would look
# wrong.
N_MOVES = 4
_FLIP_PERM = (0, 3, 2, 1)
TRANSFORMS = [(k, f) for f in (False, True) for k in range(4)]


def action_permutation(k: int, flip: bool) -> np.ndarray:
    """perm[a] is what action `a` becomes under (flip x, then rot90 k times)."""
    # int64 on every platform; actions in sample() is torch.int64.
    perm = np.arange(N_ACTIONS, dtype=np.int64)
    for a in range(N_MOVES):
        perm[a] = ((_FLIP_PERM[a] if flip else a) + k) % N_MOVES
    return perm


_PERM = {t: torch.from_numpy(action_permutation(*t)) for t in TRANSFORMS}
# next_legal is a mask indexed by action, so it moves by the INVERSE
# permutation: the entry for the new action perm[a] is the old entry for a.
_INV = {t: torch.from_numpy(np.argsort(action_permutation(*t))) for t in TRANSFORMS}


def transform_board(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    if flip:
        x = x.flip(-2)
    return torch.rot90(x, k, dims=(-2, -1)) if k else x


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
    """Uniform experience replay, storing states as uint8.

    Five of the six channels are binary and the sixth is a 0..1 danger level, so
    float32 storage wastes three quarters of the memory for no precision that
    matters. Quantising to uint8 takes a 50,000-transition buffer from 406 MB to
    101 MB, which is what makes running several training runs in parallel
    comfortable rather than a memory-pressure problem.

    Prioritised replay is a planned upgrade. Uniform first, so the prioritised
    version has an honest baseline to be measured against rather than being
    adopted on faith.
    """

    SCALE = 255.0

    def __init__(self, capacity: int, rng: random.Random = None, augment: bool = False):
        self.memory = deque(maxlen=capacity)
        self.rng = rng or random.Random()
        # Off by default so the un-augmented runs already in experiments/runs/
        # remain the baseline this is measured against.
        self.augment = augment

    @classmethod
    def _pack(cls, state):
        if state is None:
            return None
        return (np.clip(state, 0.0, 1.0) * cls.SCALE).astype(np.uint8)

    def push(self, state, action, next_state, reward, done, next_legal=None):
        if next_legal is None:
            next_legal = np.ones(N_ACTIONS, dtype=bool)
        self.memory.append(
            Transition(self._pack(state), action, self._pack(next_state), reward,
                       done, np.asarray(next_legal, dtype=bool))
        )

    def sample(self, batch_size: int):
        batch = self.rng.sample(self.memory, batch_size)

        actions = torch.tensor([t.action for t in batch], dtype=torch.int64)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32)

        states = torch.from_numpy(np.stack([t.state for t in batch])).float().div_(self.SCALE)

        # Terminal transitions have no successor. Feed a zero state and mask it
        # out with `dones` so the shapes stay rectangular.
        zero = np.zeros_like(batch[0].state)
        next_states = torch.from_numpy(
            np.stack([t.next_state if t.next_state is not None else zero for t in batch])
        ).float().div_(self.SCALE)

        next_legal = torch.from_numpy(np.stack([t.next_legal for t in batch]))

        if self.augment:
            states, next_states, actions, next_legal = self._augment(
                states, next_states, actions, next_legal)

        return states, actions, next_states, rewards, dones, next_legal

    def _augment(self, states, next_states, actions, next_legal):
        """Apply an independent random symmetry to each sample in the batch.

        Per-sample rather than per-batch: one orientation for a whole batch
        would correlate every gradient in it. Done by grouping, so it costs
        eight small tensor ops rather than a Python loop over the batch.

        Rotations need a square board. The 17x17 arena and the 13x13 egocentric
        window both are, but a non-square view would silently rotate into the
        wrong shape, so it is checked rather than assumed.
        """
        if states.shape[-1] != states.shape[-2]:
            return states, next_states, actions, next_legal

        n = states.shape[0]
        pick = torch.tensor([self.rng.randrange(len(TRANSFORMS)) for _ in range(n)])
        for i, t in enumerate(TRANSFORMS):
            if t == (0, False):
                continue
            idx = (pick == i).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            k, flip = t
            states[idx] = transform_board(states[idx], k, flip)
            next_states[idx] = transform_board(next_states[idx], k, flip)
            actions[idx] = _PERM[t][actions[idx]]
            next_legal[idx] = next_legal[idx][:, _INV[t]]
        return states, next_states, actions, next_legal

    def __len__(self):
        return len(self.memory)
