"""Training code for the DQN agent. Not imported during official games.

Double DQN with a target network and uniform experience replay, plus optional
potential-based reward shaping.

Every hyperparameter can be overridden from the environment, e.g.

    DQN_LR=0.0002 DQN_SHAPING=0 python main.py play --no-gui \
        --agents dqn_agent --train 1 --scenario coin-heaven --n-rounds 1500

so that sweeps and ablations are one shell loop rather than a source edit. Each
run appends a row per round to training_log.csv for the report's figures.
"""

import csv
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

import events as e
import settings as s

from .callbacks import (MODEL_FILE, SAFE, blast_coords, choice_mask, escape_exists,
                        save_model, state_to_features, steps_until_lethal, view_shape)
from .model import ACTIONS, N_CHANNELS, QNetwork, ReplayBuffer, masked_argmax

LOG_FILE = Path(__file__).parent / "training_log.csv"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def _env(name: str, default, cast=float):
    return cast(os.environ.get(name, default))


CONFIG = {
    "gamma": _env("DQN_GAMMA", 0.95),
    "lr": _env("DQN_LR", 5e-4),
    "batch_size": _env("DQN_BATCH", 128, int),
    "buffer_size": _env("DQN_BUFFER", 50_000, int),
    "learn_start": _env("DQN_LEARN_START", 2_000, int),
    # batch 128 every 16 steps consumes the same 8 samples per environment step
    # as the old batch 32 every 4, but pays the per-update Python and BLAS
    # overhead a quarter as often. The replay ratio is what matters for sample
    # efficiency; the update frequency is mostly a speed knob.
    "train_every": _env("DQN_TRAIN_EVERY", 16, int),
    "target_update": _env("DQN_TARGET_UPDATE", 1_000, int),
    "eps_start": _env("DQN_EPS_START", 1.0),
    "eps_end": _env("DQN_EPS_END", 0.05),
    "eps_decay_rounds": _env("DQN_EPS_DECAY", 400, int),
    "shaping": _env("DQN_SHAPING", 1, int),
    "shaping_scale": _env("DQN_SHAPING_SCALE", 0.05),
    # Discount used INSIDE the shaping term, deliberately separate from gamma.
    # At 0.95 the term gamma*Phi(s') - Phi(s) leaves a residual of
    # Phi(s)*(gamma-1) = +0.005*distance every step the distance does not
    # change, so shuffling between two tiles far from any coin paid +0.065 per
    # two steps forever. At 1.0 standing still is worth exactly 0 and the step
    # penalty makes it strictly negative. See experiments/LOG.md 2026-09-05.
    "shaping_gamma": _env("DQN_SHAPING_GAMMA", 1.0),
    # Distance beyond which the potential flattens. Without a cap, collecting a
    # coin when the next one is far is a large negative jump in Phi: at scale
    # 0.1 uncapped, collecting could cost -1.77 against a +1.0 coin, so the
    # agent was penalised for finishing a round.
    "shaping_cap": _env("DQN_SHAPING_CAP", 10, int),
    "grad_clip": _env("DQN_GRAD_CLIP", 10.0),
    "save_every": _env("DQN_SAVE_EVERY", 50, int),
    # Numbered snapshots so an honest learning curve can be built afterwards
    # by replaying each one frozen. 0 disables.
    "checkpoint_every": _env("DQN_CHECKPOINT_EVERY", 0, int),
    "seed": _env("DQN_SEED", -1, int),
}

# Custom events. The spec is explicit that a dense reward signal beats a sparse
# one and that "the computational burden of adding rewards is usually not as
# great as that of adding features", so bomb behaviour is taught through events
# rather than through hand-built state features.
GOOD_BOMB = "GOOD_BOMB"                  # will break crates or catch an opponent, and escapable
USELESS_BOMB = "USELESS_BOMB"            # breaks nothing, hits nobody
SUICIDAL_BOMB = "SUICIDAL_BOMB"          # no escape route exists from where it was dropped
ESCAPED_DANGER = "ESCAPED_DANGER"        # was standing somewhere lethal, now is not
STAYED_IN_DANGER = "STAYED_IN_DANGER"    # was in a blast and still is
ENTERED_DANGER = "ENTERED_DANGER"        # walked into a blast that was not threatening it before
CLOSER_TO_SAFETY = "CLOSER_TO_SAFETY"    # in danger, and the blast now goes off later than before

REWARDS = {
    e.COIN_COLLECTED: 1.0,
    e.INVALID_ACTION: -0.5,
    e.WAITED: -0.05,
    e.CRATE_DESTROYED: 0.3,
    e.COIN_FOUND: 0.2,
    e.KILLED_OPPONENT: 5.0,
    e.KILLED_SELF: -6.0,
    e.GOT_KILLED: -6.0,

    # Bombs. Paired so the opposites cancel rather than leaving a farmable
    # positive, which is the trap the spec warns about and which bit us once
    # already (LOG.md 2026-09-05).
    GOOD_BOMB: 0.4,
    USELESS_BOMB: -0.3,
    SUICIDAL_BOMB: -3.0,     # heavy: this is the single biggest failure mode
    ESCAPED_DANGER: 0.5,
    STAYED_IN_DANGER: -0.4,
    ENTERED_DANGER: -0.6,
    CLOSER_TO_SAFETY: 0.15,
}
STEP_PENALTY = -0.02


def _bomb_events(self, old_state, action, new_state, events):
    """Judge the step's bomb behaviour and append the matching custom events.

    Everything here is a function of the two game states, not of the action
    taken to reach them, except the bomb verdict itself which needs to know a
    bomb was dropped. That keeps it close to the spec's advice that auxiliary
    rewards should depend on states.
    """
    extra = []
    field = old_state["field"]
    ox, oy = old_state["self"][3]

    if e.BOMB_DROPPED in events:
        hit = blast_coords(field, ox, oy)
        crates = sum(1 for (x, y) in hit if field[x, y] == 1)
        opponents = sum(1 for _, _, _, pos in old_state["others"] if tuple(pos) in hit)

        if not escape_exists(old_state, extra_bomb=(ox, oy)):
            extra.append(SUICIDAL_BOMB)
        elif crates or opponents:
            extra.append(GOOD_BOMB)
        else:
            extra.append(USELESS_BOMB)

    if new_state is None:
        return extra

    was = int(steps_until_lethal(old_state)[ox, oy])
    nx, ny = new_state["self"][3]
    now = int(steps_until_lethal(new_state)[nx, ny])

    in_danger_before = was < SAFE
    in_danger_now = now < SAFE

    if in_danger_before and not in_danger_now:
        extra.append(ESCAPED_DANGER)
    elif not in_danger_before and in_danger_now:
        extra.append(ENTERED_DANGER)
    elif in_danger_before and in_danger_now:
        # Both counts are "steps until this tile kills me". A larger number is
        # further from death, so moving down a blast toward its edge counts as
        # progress even before the agent is fully clear.
        extra.append(CLOSER_TO_SAFETY if now > was else STAYED_IN_DANGER)

    return extra


def setup_training(self):
    """Called once after setup(), only in training mode."""
    cfg = CONFIG
    if cfg["seed"] >= 0:
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])

    self.cfg = cfg
    self.buffer = ReplayBuffer(cfg["buffer_size"], rng=random.Random(cfg["seed"] if cfg["seed"] >= 0 else None))

    rows, cols = view_shape(s.ROWS, s.COLS, self.view)
    self.target_network = QNetwork(rows=rows, cols=cols, n_channels=N_CHANNELS).to(self.device)
    self.target_network.load_state_dict(self.q_network.state_dict())
    self.target_network.eval()

    self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=cfg["lr"])
    self.loss_fn = nn.SmoothL1Loss()

    self.total_steps = 0
    self.grad_steps = 0
    self.epsilon = cfg["eps_start"]
    self.current_round = 0
    self.recent_losses = deque(maxlen=500)
    _reset_round_stats(self)

    _open_log()

    self.logger.info(f"Training config: {cfg}")


LOG_COLUMNS = [
    "round", "steps", "score", "coins", "invalid_actions", "waited",
    "crates", "good_bombs", "useless_bombs", "suicidal_bombs",
    "reward_sum", "epsilon", "mean_loss", "buffer",
]


def _open_log():
    """Start a log the rows will actually line up with.

    The header used to be written only when the file was absent, so adding
    columns meant appending wider rows under a narrower header and silently
    corrupting the file. If the existing header does not match, the old log is
    moved aside rather than overwritten: it is somebody's experiment.
    """
    if LOG_FILE.exists():
        with open(LOG_FILE) as fh:
            header = fh.readline().strip().split(",")
        if header == LOG_COLUMNS:
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        LOG_FILE.rename(LOG_FILE.with_name(f"training_log.{stamp}.csv"))

    with open(LOG_FILE, "w", newline="") as fh:
        csv.writer(fh).writerow(LOG_COLUMNS)


def _reset_round_stats(self):
    self.round_reward = 0.0
    self.round_coins = 0
    self.round_invalid = 0
    self.round_waited = 0
    self.round_steps = 0
    self.round_crates = 0
    self.round_good_bombs = 0
    self.round_useless_bombs = 0
    self.round_suicidal_bombs = 0


def _tally(self, events, reward):
    """Per-round counters. The bomb columns are what tells you whether task 2 is
    working: suicidal bombs must fall to ~0 and crates must rise."""
    self.round_steps += 1
    self.round_reward += reward
    self.round_coins += events.count(e.COIN_COLLECTED)
    self.round_invalid += events.count(e.INVALID_ACTION)
    self.round_waited += events.count(e.WAITED)
    self.round_crates += events.count(e.CRATE_DESTROYED)
    self.round_good_bombs += events.count(GOOD_BOMB)
    self.round_useless_bombs += events.count(USELESS_BOMB)
    self.round_suicidal_bombs += events.count(SUICIDAL_BOMB)


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    """Called once per step, after the consequences of the action are known."""
    if old_game_state is None or self_action is None:
        return

    if new_game_state["round"] != self.current_round:
        self.current_round = new_game_state["round"]
        _reset_round_stats(self)
        _update_epsilon(self)

    events = events + _bomb_events(self, old_game_state, self_action, new_game_state, events)

    old_features, phi_old = _encode(self, old_game_state)
    new_features, phi_new = _encode(self, new_game_state)
    reward = _reward(self, events, phi_old, phi_new)

    self.buffer.push(old_features, ACTIONS.index(self_action), new_features, reward,
                     False, choice_mask(self, new_game_state).numpy())
    _learn(self)

    _tally(self, events, reward)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Called once per round, after the agent's final step."""
    # A round that merely ran out of steps is NOT a terminal state, it is a
    # truncated one (Pardo et al. 2018). Storing it with done=True would teach
    # the agent that the world ends at step 400, and would hand it the shaping
    # bonus -Phi(s) for simply being far from a coin when the clock stopped.
    # Only a real death terminates.
    died = e.KILLED_SELF in events or e.GOT_KILLED in events

    if last_game_state is not None and last_action is not None:
        events = events + _bomb_events(self, last_game_state, last_action, None, events)
        last_features, phi_last = _encode(self, last_game_state)
        reward = _reward(self, events, phi_last, 0.0 if died else phi_last)
        if died:
            self.buffer.push(last_features, ACTIONS.index(last_action), None, reward, True)
            _learn(self)

        _tally(self, events, reward)

    score = last_game_state["self"][1] if last_game_state else 0
    mean_loss = float(np.mean(self.recent_losses)) if self.recent_losses else 0.0

    with open(LOG_FILE, "a", newline="") as fh:
        csv.writer(fh).writerow([
            self.current_round, self.round_steps, score, self.round_coins,
            self.round_invalid, self.round_waited, self.round_crates,
            self.round_good_bombs, self.round_useless_bombs, self.round_suicidal_bombs,
            round(self.round_reward, 3),
            round(self.epsilon, 4), round(mean_loss, 5), len(self.buffer),
        ])

    self.logger.info(
        f"Round {self.current_round}: score {score}, coins {self.round_coins}, "
        f"crates {self.round_crates}, bombs good/useless/suicidal "
        f"{self.round_good_bombs}/{self.round_useless_bombs}/{self.round_suicidal_bombs}, "
        f"eps {self.epsilon:.3f}, loss {mean_loss:.4f}"
    )

    if self.current_round % self.cfg["save_every"] == 0:
        _save(self)

    every = self.cfg["checkpoint_every"]
    if every and self.current_round % every == 0:
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        path = CHECKPOINT_DIR / f"dqn-r{self.current_round:06d}.pt"
        save_model(path, self.q_network, self.view)
        self.logger.info(f"Checkpoint {path.name}")

    _reset_round_stats(self)


def _save(self):
    save_model(MODEL_FILE, self.q_network, self.view)
    self.logger.info(f"Saved model to {MODEL_FILE.name}")


def _update_epsilon(self):
    cfg = self.cfg
    frac = min(1.0, self.current_round / max(1, cfg["eps_decay_rounds"]))
    self.epsilon = cfg["eps_start"] + frac * (cfg["eps_end"] - cfg["eps_start"])


def _learn(self):
    """One Double DQN gradient step, throttled by train_every."""
    cfg = self.cfg
    self.total_steps += 1
    if len(self.buffer) < cfg["learn_start"] or self.total_steps % cfg["train_every"] != 0:
        return

    states, actions, next_states, rewards, dones, next_legal = self.buffer.sample(cfg["batch_size"])

    self.q_network.train()
    q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Double DQN: the online network picks the successor action, the target
        # network scores it. Decoupling these is what stops the systematic
        # over-estimation that plain DQN suffers from.
        # The mask has to be applied here too. Actions the agent is never
        # allowed to take are never trained, so their Q values are noise; left
        # unmasked they would win the argmax and poison every target.
        best_next = masked_argmax(self.q_network(next_states), next_legal).unsqueeze(1)
        next_q = self.target_network(next_states).gather(1, best_next).squeeze(1)
        targets = rewards + cfg["gamma"] * next_q * (1.0 - dones)

    loss = self.loss_fn(q_values, targets)
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.q_network.parameters(), cfg["grad_clip"])
    self.optimizer.step()
    self.q_network.eval()

    self.recent_losses.append(loss.item())
    self.grad_steps += 1
    if self.grad_steps % cfg["target_update"] == 0:
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.logger.debug(f"Target network synced at {self.grad_steps} gradient steps")


def _encode(self, game_state: dict):
    """Board tensor and BFS potential for one state.

    This used to memoise on (round, step) to avoid encoding each state twice.
    That was catastrophically wrong: environment.do_step increments self.step
    ONCE at the top, so the state handed to act() and the state handed to
    game_events_occurred() as its successor carry the SAME step number. The
    cache therefore returned the predecessor's tensor for every successor, and
    100% of transitions in the replay buffer had next_state == state.

    The Bellman target degenerates to Q(s,a) <- r + gamma * max_a' Q(s,a'),
    which can only ever learn immediate rewards. It explained the whole failure:
    wall avoidance (a one-step penalty) was learned perfectly while coin
    collection (multi-step) was impossible, and all six sweep configs were
    identical because they all trained on the same corrupted transitions.

    Profiling put this encoding at ~2% of a training round, so the cache was
    never worth its risk. Do not reintroduce it.
    """
    if game_state is None:
        return None, 0.0
    return state_to_features(game_state, self.view), _potential(self, game_state)


def _reward(self, events: List[str], phi_old: float, phi_new: float) -> float:
    reward = sum(REWARDS.get(ev, 0.0) for ev in events) + STEP_PENALTY
    if self.cfg["shaping"]:
        # Potential-based shaping, F = gamma_s * Phi(s') - Phi(s). Ng, Harada &
        # Russell (1999): a term of this form leaves the optimal policy
        # unchanged, which is why the potential depends only on the state and
        # never on the action taken to reach it.
        #
        # gamma_s is 1.0 by default rather than the agent's gamma. That trades
        # the theorem's exact invariance for a term with no per-step drift,
        # which matters far more in practice: the drift version handed the agent
        # a dense positive reward for making no progress and it took it.
        reward += self.cfg["shaping_gamma"] * phi_new - phi_old
    return reward


def _potential(self, game_state: dict) -> float:
    if game_state is None:
        return 0.0
    distance = _distance_to_nearest_coin(game_state)
    if distance is None:
        return 0.0
    return -self.cfg["shaping_scale"] * min(distance, self.cfg["shaping_cap"])


def _distance_to_nearest_coin(game_state: dict):
    """Breadth-first search over free tiles. Training-only, so its cost never
    counts against the 0.5 s per-step budget in official games."""
    coins = game_state["coins"]
    if not coins:
        return None

    field = game_state["field"]
    _, _, _, start = game_state["self"]
    targets = set(map(tuple, coins))

    visited = np.zeros_like(field, dtype=bool)
    visited[start] = True
    queue = deque([(start, 0)])

    while queue:
        (x, y), dist = queue.popleft()
        if (x, y) in targets:
            return dist
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue
            if visited[nx, ny] or field[nx, ny] != 0:
                continue
            visited[nx, ny] = True
            queue.append(((nx, ny), dist + 1))

    return None  # no coin reachable
