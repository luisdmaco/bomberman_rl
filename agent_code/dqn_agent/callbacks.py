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

# DQN_MODEL_FILE lets the evaluation pass point at one specific checkpoint
# without disturbing the live training file.
MODEL_FILE = Path(os.environ.get("DQN_MODEL_FILE") or (Path(__file__).parent / "dqn-model.pt"))

# "global" feeds the whole board with the agent as one lit pixel, so the network
# has to learn "walk toward a coin" separately for every position on the map.
# "ego:N" re-centres an N x N window on the agent, which makes the same local
# pattern mean the same thing everywhere and turns the task translation
# invariant. Kept switchable so the report can ablate it.
# Sentinel for "no bomb threatens this tile".
SAFE = 99

VIEW = os.environ.get("DQN_VIEW", "ego:13")


def view_shape(rows: int, cols: int, view: str = None):
    view = view or VIEW
    if view.startswith("ego:"):
        size = int(view.split(":")[1])
        return size, size
    return rows, cols


def save_model(path, network, view: str):
    """Store the view alongside the weights.

    The view decides the network's input shape, so weights saved under one view
    cannot be loaded under another. Reading it from an environment variable was
    a submission bug waiting to happen: official games set no environment, so a
    global-view model would silently be given an egocentric architecture and
    fail to load. Now the checkpoint describes itself.
    """
    torch.save({"view": view, "state_dict": network.state_dict()}, path)


def peek_view(path, fallback: str) -> str:
    """What view was this checkpoint trained with? Falls back for older files
    that hold a bare state_dict."""
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return fallback
    if isinstance(blob, dict) and "view" in blob:
        return blob["view"]
    return fallback


def load_weights(path, network, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    network.load_state_dict(blob["state_dict"] if "state_dict" in blob else blob)


def setup(self):
    """Called once before the first round."""
    # In official games the agent gets one thread. Pinning it here keeps the
    # measured per-step time honest and avoids thread-thrash on a loaded box.
    #
    # During training the default is also 1, deliberately. This network is small
    # enough that a second thread buys under 10%, while a second *process* buys
    # a whole extra run. One thread each and several runs in parallel beats one
    # run with many threads. Override with DQN_THREADS if you disagree.
    torch.set_num_threads(1 if not self.train else int(os.environ.get("DQN_THREADS", 1)))

    self.device = torch.device("cpu")

    resume = os.environ.get("DQN_RESUME", "0") == "1"
    will_load = (not self.train or resume) and MODEL_FILE.is_file()
    # Ask the checkpoint what shape it needs before building anything.
    self.view = peek_view(MODEL_FILE, VIEW) if will_load else VIEW

    rows, cols = view_shape(s.ROWS, s.COLS, self.view)
    self.q_network = QNetwork(rows=rows, cols=cols, n_channels=N_CHANNELS).to(self.device)

    # DQN_ACTIONS restricts the action space for a curriculum stage, e.g.
    # DQN_ACTIONS=UP,RIGHT,DOWN,LEFT for task 1. Defaults to the full set, so
    # the tournament agent is never silently restricted.
    allowed = os.environ.get("DQN_ACTIONS", ",".join(ACTIONS)).split(",")
    self.allowed_actions = [a.strip() for a in allowed if a.strip()]
    self.action_mask = action_mask(self.allowed_actions)

    if will_load:
        load_weights(MODEL_FILE, self.q_network, self.device)
        self.logger.info(f"Loaded model from {MODEL_FILE.name} (view {self.view})")
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

    # Remove moves the game will reject anyway (walking into a wall, a crate,
    # another agent or a bomb). This is not the shortcut the spec forbids: it
    # never says which move is best, it only drops moves that are not moves.
    # It takes invalid actions to zero and frees the network from spending
    # capacity on a rule the environment already enforces. Set DQN_LEGAL_MASK=0
    # for the ablation.
    self.legal_mask = os.environ.get("DQN_LEGAL_MASK", "1") == "1"

    # Probability that a random exploratory action is BOMB. Uniform over six
    # actions would make it 1/6, and on a crate-dense board that kills the agent
    # within a few steps, so rounds end before any experience accumulates. The
    # framework's own template agent does the same thing (p=[.2,.2,.2,.2,.1,.1]).
    # This shapes exploration only; the greedy policy is untouched.
    self.explore_bomb_p = float(os.environ.get("DQN_EXPLORE_BOMB", 0.10))


def legal_actions(game_state: dict) -> torch.Tensor:
    """Boolean mask over ACTIONS: which ones the environment would actually
    carry out. Mirrors environment.perform_agent_action and tile_is_free."""
    field = game_state["field"]
    _, _, bombs_left, (x, y) = game_state["self"]

    blocked = {(bx, by) for (bx, by), _ in game_state["bombs"]}
    blocked |= {pos for _, _, _, pos in game_state["others"]}

    mask = torch.zeros(len(ACTIONS), dtype=torch.bool)
    for name, (dx, dy) in (("UP", (0, -1)), ("RIGHT", (1, 0)),
                           ("DOWN", (0, 1)), ("LEFT", (-1, 0))):
        nx, ny = x + dx, y + dy
        if field[nx, ny] == 0 and (nx, ny) not in blocked:
            mask[ACTIONS.index(name)] = True
    mask[ACTIONS.index("WAIT")] = True
    if bombs_left:
        mask[ACTIONS.index("BOMB")] = True
    return mask


def choice_mask(self, game_state: dict) -> torch.Tensor:
    """The curriculum's allowed actions, narrowed to what is legal here."""
    mask = self.action_mask
    if getattr(self, "legal_mask", False):
        narrowed = mask & legal_actions(game_state)
        # Boxed in with every allowed move blocked: WAIT rather than pick a
        # move the environment will reject.
        mask = narrowed if narrowed.any() else torch.zeros_like(mask).index_fill_(
            0, torch.tensor([ACTIONS.index("WAIT")]), True)
    return mask


def act(self, game_state: dict) -> str:
    """Pick an action for the current step."""
    if game_state is None:
        return "WAIT"

    mask = choice_mask(self, game_state)

    epsilon = getattr(self, "epsilon", 0.0) if self.train else getattr(self, "play_epsilon", 0.0)
    if np.random.rand() < epsilon:
        idx = [i for i in range(len(ACTIONS)) if mask[i]]
        weights = np.array([self.explore_bomb_p if ACTIONS[i] == "BOMB" else 1.0
                            for i in idx], dtype=float)
        weights /= weights.sum()
        action = ACTIONS[int(np.random.choice(idx, p=weights))]
        self.logger.debug(f"Exploring: {action}")
        return action

    features = state_to_features(game_state, self.view)
    with torch.no_grad():
        tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        q_values = self.q_network(tensor).squeeze(0)

    action = ACTIONS[int(masked_argmax(q_values, mask).item())]
    self.logger.debug(f"Q values {q_values.tolist()} -> {action}")
    return action


def blast_coords(field, bx, by):
    """Tiles a bomb at (bx, by) will hit. Stops at stone walls ONLY: the blast
    passes straight through crates, which is the most common modelling mistake
    in this framework."""
    coords = [(bx, by)]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for step in range(1, s.BOMB_POWER + 1):
            x, y = bx + dx * step, by + dy * step
            if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]):
                break
            if field[x, y] == -1:
                break
            coords.append((x, y))
    return coords


def steps_until_lethal(game_state: dict, extra_bomb=None) -> np.ndarray:
    """For every tile, how many steps until standing there kills you.

    LETHAL_NOW (0) means the tile kills at the end of THIS step; moving off it
    now still saves you. SAFE is a large sentinel.

    Timings measured against a live game, not inferred (see experiments/LOG.md,
    C1 on 2026-09-08):

      step 1  drop            state shows no bomb yet
      step 2  timer 3         explosion_map 0
      step 3  timer 2         explosion_map 0
      step 4  timer 1         explosion_map 0
      step 5  timer 0         detonates at the end of this step, KILLED_SELF
      step 6  no bomb         explosion_map 1, still lethal at end of step
      step 7  no bomb         explosion_map 0, safe

    So a bomb showing timer T is lethal at the end of the step T steps from now,
    and `explosion_map > 0` means lethal at the end of this step. An agent that
    sees timer T has T+1 actions (this one included) to get clear.
    """
    field = game_state["field"]
    out = np.full(field.shape, SAFE, dtype=np.int16)

    out[game_state["explosion_map"] > 0] = 0

    bombs = list(game_state["bombs"])
    if extra_bomb is not None:
        bombs.append((extra_bomb, s.BOMB_TIMER - 1))

    for (bx, by), timer in bombs:
        for (x, y) in blast_coords(field, bx, by):
            out[x, y] = min(out[x, y], int(timer))
    return out


def danger_map(game_state: dict) -> np.ndarray:
    """Per-tile lethality as a 0..1 channel, 1.0 = kills at the end of this step.

    Derived from steps_until_lethal, so the encoding is monotone in urgency:
    timer 3 -> 0.25, timer 2 -> 0.50, timer 1 -> 0.75, timer 0 or a live
    explosion -> 1.0.
    """
    steps = steps_until_lethal(game_state)
    danger = np.zeros(steps.shape, dtype=np.float32)
    threatened = steps < SAFE
    danger[threatened] = (s.BOMB_TIMER - steps[threatened]) / s.BOMB_TIMER
    return np.clip(danger, 0.0, 1.0)


def steps_to_safety(game_state: dict, from_pos=None, extra_bomb=None):
    """How many moves to the nearest tile that is still safe on arrival.

    0 means already safe. None means no escape exists.

    Progress out of a blast has to be measured this way, not by steps-until-
    lethal: a whole blast shares one countdown, so that number falls by exactly
    one every step no matter where the agent moves inside it. Using it made
    "getting closer to safety" impossible to detect and charged the agent for
    every step of a correct escape.
    """
    field = game_state["field"]
    start = tuple(from_pos or game_state["self"][3])
    steps = steps_until_lethal(game_state, extra_bomb=extra_bomb)

    blocked = {(bx, by) for (bx, by), _ in game_state["bombs"]}
    blocked |= {pos for _, _, _, pos in game_state["others"]}

    seen = {start}
    frontier = [(start, 0)]
    while frontier:
        (x, y), d = frontier.pop(0)
        if steps[x, y] >= SAFE:
            return d
        if d >= s.BOMB_TIMER:
            continue
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue
            if (nx, ny) in seen or field[nx, ny] != 0 or (nx, ny) in blocked:
                continue
            if steps[nx, ny] <= d:
                continue
            seen.add((nx, ny))
            frontier.append(((nx, ny), d + 1))
    return None


def escape_exists(game_state: dict, from_pos=None, extra_bomb=None) -> bool:
    """Can the agent reach a tile that will not kill it, in time?

    Walks outward from the agent's position. A tile reached after d moves is
    survivable only if it is still safe at that moment, i.e. its steps-until-
    lethal is greater than d. Used to tell a bomb worth dropping from one that
    is suicide.
    """
    return steps_to_safety(game_state, from_pos, extra_bomb) is not None


def _unused_escape_exists(game_state, from_pos=None, extra_bomb=None):
    """Kept only so the original breadth-first version stays readable."""
    field = game_state["field"]
    start = tuple(from_pos or game_state["self"][3])
    steps = steps_until_lethal(game_state, extra_bomb=extra_bomb)

    blocked = {(bx, by) for (bx, by), _ in game_state["bombs"]}
    blocked |= {pos for _, _, _, pos in game_state["others"]}

    seen = {start}
    frontier = [(start, 0)]
    budget = s.BOMB_TIMER
    while frontier:
        (x, y), d = frontier.pop(0)
        if steps[x, y] >= SAFE:
            return True
        if d >= budget:
            continue
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue
            if (nx, ny) in seen or field[nx, ny] != 0 or (nx, ny) in blocked:
                continue
            # Arriving after d+1 moves: the tile must outlast that.
            if steps[nx, ny] <= d:
                continue
            seen.add((nx, ny))
            frontier.append(((nx, ny), d + 1))
    return False


def state_to_features(game_state: dict, view: str = None) -> np.ndarray:
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

    view = view or VIEW
    if view.startswith("ego:"):
        channels = _recentre(channels, (sx, sy), int(view.split(":")[1]))

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
