"""Game-facing interface for the DQN agent.

Only this file is imported when the agent plays without training, so it must
work standalone: the official submission test runs a single game with
self.train = False and never imports train.py.
"""

import os
from pathlib import Path

import numpy as np
import torch

import settings as s

from .model import ACTIONS, N_CHANNELS, QNetwork, action_mask, masked_argmax

MODEL_FILE = Path(__file__).parent / "dqn-model.pt"

# "global" feeds the whole board with the agent as one lit pixel, so the network
# has to learn "walk toward a coin" separately for every position on the map.
# "ego:N" re-centres an N x N window on the agent, which makes the same local
# pattern mean the same thing everywhere and turns the task translation
# invariant. Kept switchable so the report can ablate it.
VIEW = os.environ.get("DQN_VIEW", "ego:13")


def view_shape(rows: int, cols: int):
    if VIEW.startswith("ego:"):
        size = int(VIEW.split(":")[1])
        return size, size
    return rows, cols


def setup(self):
    """Called once before the first round."""
    # In official games the agent gets one thread. Pinning it here keeps the
    # measured per-step time honest and avoids thread-thrash on a loaded box.
    if not self.train:
        torch.set_num_threads(1)

    self.device = torch.device("cpu")
    rows, cols = view_shape(s.ROWS, s.COLS)
    self.q_network = QNetwork(rows=rows, cols=cols, n_channels=N_CHANNELS).to(self.device)

    # DQN_ACTIONS restricts the action space for a curriculum stage, e.g.
    # DQN_ACTIONS=UP,RIGHT,DOWN,LEFT for task 1. Defaults to the full set, so
    # the tournament agent is never silently restricted.
    allowed = os.environ.get("DQN_ACTIONS", ",".join(ACTIONS)).split(",")
    self.allowed_actions = [a.strip() for a in allowed if a.strip()]
    self.action_mask = action_mask(self.allowed_actions)

    resume = os.environ.get("DQN_RESUME", "0") == "1"
    if (not self.train or resume) and MODEL_FILE.is_file():
        state_dict = torch.load(MODEL_FILE, map_location=self.device)
        self.q_network.load_state_dict(state_dict)
        self.logger.info(f"Loaded model from {MODEL_FILE.name}")
    elif not self.train:
        # Playing with no trained weights is almost certainly a mistake, so say
        # so loudly rather than silently acting on random numbers.
        self.logger.warning(f"No model file at {MODEL_FILE}. Acting on an untrained network.")
    else:
        self.logger.info("Training from scratch.")

    self.q_network.eval()
    # train.py overwrites this each round; without it we act greedily.
    self.epsilon = 0.0

    # A purely greedy policy on a board that does not change is a fixed point:
    # if the argmax at A says "right" and at B says "left", the agent shuttles
    # between them forever. A little noise at play time breaks such cycles.
    # This is a safety net, not a fix. The real fix is a Q function whose argmax
    # does not cycle, and the value here should trend to 0 as that improves.
    self.play_epsilon = float(os.environ.get("DQN_PLAY_EPSILON", 0.0))


def act(self, game_state: dict) -> str:
    """Pick an action for the current step."""
    if game_state is None:
        return "WAIT"

    epsilon = getattr(self, "epsilon", 0.0) if self.train else getattr(self, "play_epsilon", 0.0)
    if np.random.rand() < epsilon:
        action = str(np.random.choice(self.allowed_actions))
        self.logger.debug(f"Exploring: {action}")
        return action

    features = state_to_features(game_state)
    with torch.no_grad():
        tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        q_values = self.q_network(tensor).squeeze(0)

    action = ACTIONS[int(masked_argmax(q_values, self.action_mask).item())]
    self.logger.debug(f"Q values {q_values.tolist()} -> {action}")
    return action


def danger_map(game_state: dict) -> np.ndarray:
    """Per-tile lethality, 0.0 (safe) to 1.0 (about to detonate).

    NOTE: the exact bomb-timer indexing must be re-verified against
    environment.do_step before this is relied on for task 2. It is all zeros in
    coin-heaven, so gate 1 does not exercise it.
    """
    field = game_state["field"]
    danger = np.zeros_like(field, dtype=np.float32)

    # Explosions already on the board: any nonzero entry is lethal right now.
    explosion_map = game_state["explosion_map"]
    danger[explosion_map > 0] = 1.0

    for (bx, by), timer in game_state["bombs"]:
        # Urgency rises as the countdown falls. timer == 0 means it goes off at
        # the end of this step.
        urgency = float(s.BOMB_TIMER - timer) / s.BOMB_TIMER
        urgency = min(max(urgency, 0.1), 1.0)
        danger[bx, by] = max(danger[bx, by], urgency)

        # Blasts stop at stone walls only. They pass straight through crates,
        # which is the single most common modelling mistake in this framework.
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            for step in range(1, s.BOMB_POWER + 1):
                x, y = bx + dx * step, by + dy * step
                if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]):
                    break
                if field[x, y] == -1:
                    break
                danger[x, y] = max(danger[x, y], urgency)

    return danger


def state_to_features(game_state: dict) -> np.ndarray:
    """Turn the game state dict into a (N_CHANNELS, cols, rows) float array.

    No hand-crafted pathfinding or situational features here on purpose. The
    whole point of the DQN arm of this project is that the network learns its
    own features from the raw board, so that it can be compared honestly
    against the engineered-feature Q-learning agent.
    """
    if game_state is None:
        return None

    field = game_state["field"]
    channels = np.zeros((N_CHANNELS, field.shape[0], field.shape[1]), dtype=np.float32)

    channels[0][field == -1] = 1.0          # stone walls
    channels[1][field == 1] = 1.0           # crates

    _, _, _, (sx, sy) = game_state["self"]
    channels[2][sx, sy] = 1.0               # own position

    for _, _, _, (ox, oy) in game_state["others"]:
        channels[3][ox, oy] = 1.0           # opponents

    for (cx, cy) in game_state["coins"]:
        channels[4][cx, cy] = 1.0           # collectable coins

    channels[5] = danger_map(game_state)

    if VIEW.startswith("ego:"):
        channels = _recentre(channels, (sx, sy), int(VIEW.split(":")[1]))

    return channels


def _recentre(channels: np.ndarray, position, size: int) -> np.ndarray:
    """Crop a size x size window centred on the agent.

    Everything beyond the board edge is padded as stone wall, which is what it
    effectively is: not somewhere the agent can ever go.
    """
    radius = size // 2
    pad = ((0, 0), (radius, radius), (radius, radius))
    padded = np.pad(channels, pad, mode="constant", constant_values=0.0)
    padded[0] = np.pad(channels[0], pad[1:], mode="constant", constant_values=1.0)

    x, y = position
    return padded[:, x:x + size, y:y + size]
