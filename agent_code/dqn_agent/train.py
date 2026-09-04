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
from collections import deque
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

import events as e
import settings as s

from .callbacks import MODEL_FILE, state_to_features, view_shape
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

REWARDS = {
    e.COIN_COLLECTED: 1.0,
    e.INVALID_ACTION: -0.5,
    e.WAITED: -0.2,
    e.BOMB_DROPPED: -0.5,   # useless and lethal in coin-heaven
    e.KILLED_SELF: -5.0,
    e.GOT_KILLED: -5.0,
    e.CRATE_DESTROYED: 0.0,
    e.KILLED_OPPONENT: 5.0,
}
STEP_PENALTY = -0.02


def setup_training(self):
    """Called once after setup(), only in training mode."""
    cfg = CONFIG
    if cfg["seed"] >= 0:
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])

    self.cfg = cfg
    self.buffer = ReplayBuffer(cfg["buffer_size"], rng=random.Random(cfg["seed"] if cfg["seed"] >= 0 else None))

    rows, cols = view_shape(s.ROWS, s.COLS)
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
    # Consecutive steps share a state: this step's "old" is last step's "new".
    # Caching the encoding and the BFS potential halves both per step.
    self._encode_cache = {}
    _reset_round_stats(self)

    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="") as fh:
            csv.writer(fh).writerow([
                "round", "steps", "score", "coins", "invalid_actions", "waited",
                "reward_sum", "epsilon", "mean_loss", "buffer",
            ])

    self.logger.info(f"Training config: {cfg}")


def _reset_round_stats(self):
    self.round_reward = 0.0
    self.round_coins = 0
    self.round_invalid = 0
    self.round_waited = 0
    self.round_steps = 0


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    """Called once per step, after the consequences of the action are known."""
    if old_game_state is None or self_action is None:
        return

    if new_game_state["round"] != self.current_round:
        self.current_round = new_game_state["round"]
        _reset_round_stats(self)
        _update_epsilon(self)

    old_features, phi_old = _encode(self, old_game_state)
    new_features, phi_new = _encode(self, new_game_state)
    reward = _reward(self, events, phi_old, phi_new)

    self.buffer.push(old_features, ACTIONS.index(self_action), new_features, reward, False)
    _learn(self)

    self.round_steps += 1
    self.round_reward += reward
    self.round_coins += events.count(e.COIN_COLLECTED)
    self.round_invalid += events.count(e.INVALID_ACTION)
    self.round_waited += events.count(e.WAITED)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Called once per round, after the agent's final step."""
    # A round that merely ran out of steps is NOT a terminal state, it is a
    # truncated one (Pardo et al. 2018). Storing it with done=True would teach
    # the agent that the world ends at step 400, and would hand it the shaping
    # bonus -Phi(s) for simply being far from a coin when the clock stopped.
    # Only a real death terminates.
    died = e.KILLED_SELF in events or e.GOT_KILLED in events

    if last_game_state is not None and last_action is not None:
        last_features, phi_last = _encode(self, last_game_state)
        reward = _reward(self, events, phi_last, 0.0 if died else phi_last)
        if died:
            self.buffer.push(last_features, ACTIONS.index(last_action), None, reward, True)
            _learn(self)

        self.round_steps += 1
        self.round_reward += reward
        self.round_coins += events.count(e.COIN_COLLECTED)
        self.round_invalid += events.count(e.INVALID_ACTION)
        self.round_waited += events.count(e.WAITED)

    score = last_game_state["self"][1] if last_game_state else 0
    mean_loss = float(np.mean(self.recent_losses)) if self.recent_losses else 0.0

    with open(LOG_FILE, "a", newline="") as fh:
        csv.writer(fh).writerow([
            self.current_round, self.round_steps, score, self.round_coins,
            self.round_invalid, self.round_waited, round(self.round_reward, 3),
            round(self.epsilon, 4), round(mean_loss, 5), len(self.buffer),
        ])

    self.logger.info(
        f"Round {self.current_round}: score {score}, coins {self.round_coins}, "
        f"invalid {self.round_invalid}, eps {self.epsilon:.3f}, loss {mean_loss:.4f}"
    )

    if self.current_round % self.cfg["save_every"] == 0:
        _save(self)

    every = self.cfg["checkpoint_every"]
    if every and self.current_round % every == 0:
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        path = CHECKPOINT_DIR / f"dqn-r{self.current_round:06d}.pt"
        torch.save(self.q_network.state_dict(), path)
        self.logger.info(f"Checkpoint {path.name}")

    _reset_round_stats(self)


def _save(self):
    torch.save(self.q_network.state_dict(), MODEL_FILE)
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

    states, actions, next_states, rewards, dones = self.buffer.sample(cfg["batch_size"])

    self.q_network.train()
    q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Double DQN: the online network picks the successor action, the target
        # network scores it. Decoupling these is what stops the systematic
        # over-estimation that plain DQN suffers from.
        # The mask has to be applied here too. Actions the agent is never
        # allowed to take are never trained, so their Q values are noise; left
        # unmasked they would win the argmax and poison every target.
        best_next = masked_argmax(self.q_network(next_states), self.action_mask).unsqueeze(1)
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
    """Board tensor and BFS potential for one state, memoised across the two
    calls that see it (as this step's successor, then next step's predecessor)."""
    if game_state is None:
        return None, 0.0

    key = (game_state["round"], game_state["step"])
    hit = self._encode_cache.get(key)
    if hit is not None:
        return hit

    value = (state_to_features(game_state), _potential(self, game_state))
    if len(self._encode_cache) > 2:
        self._encode_cache.clear()
    self._encode_cache[key] = value
    return value


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
