#!/usr/bin/env python3
"""Where does a training round actually spend its time?

Answers the practical question: is training slow because of the neural network,
or because of the Python game loop around it? The answer decides whether GPU
acceleration is worth anything, or whether the win is elsewhere.

    python tools/profile_step.py
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import settings as s  # noqa: E402
from agent_code.dqn_agent.callbacks import state_to_features, view_shape  # noqa: E402
from agent_code.dqn_agent.model import N_CHANNELS, QNetwork, ReplayBuffer  # noqa: E402


def fake_state(n_coins=50):
    """A coin-heaven-like board, good enough for timing."""
    field = np.zeros((s.COLS, s.ROWS), dtype=int)
    field[:1, :] = field[-1:, :] = field[:, :1] = field[:, -1:] = -1
    for x in range(s.COLS):
        for y in range(s.ROWS):
            if (x + 1) * (y + 1) % 2 == 1:
                field[x, y] = -1
    free = [(x, y) for x in range(s.COLS) for y in range(s.ROWS) if field[x, y] == 0]
    rng = np.random.default_rng(0)
    picks = rng.choice(len(free), size=n_coins + 1, replace=False)
    coins = [free[i] for i in picks[:n_coins]]
    me = free[picks[-1]]
    return {
        "round": 1, "step": 1, "field": field, "bombs": [],
        "explosion_map": np.zeros_like(field), "coins": coins,
        "self": ("me", 0, True, me), "others": [], "user_input": None,
    }


def bench(label, fn, n):
    fn()  # warm up
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    per = (time.perf_counter() - t0) / n
    print(f"{label:<38}{per * 1000:8.3f} ms")
    return per


def main():
    threads = torch.get_num_threads()
    print(f"torch threads: {threads}\n")

    state = fake_state()
    rows, cols = view_shape(s.ROWS, s.COLS)
    net = QNetwork(rows=rows, cols=cols, n_channels=N_CHANNELS)
    net.eval()
    target = QNetwork(rows=rows, cols=cols, n_channels=N_CHANNELS)
    target.eval()
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    loss_fn = torch.nn.SmoothL1Loss()

    feats = state_to_features(state)
    tensor = torch.from_numpy(feats).float().unsqueeze(0)

    print("PER ENVIRONMENT STEP (every step pays these)")
    t_feat = bench("state_to_features", lambda: state_to_features(state), 400)

    def forward():
        with torch.no_grad():
            net(tensor)
    t_fwd = bench("network forward, batch 1 (act)", forward, 400)

    # BFS potential, training only
    from agent_code.dqn_agent.train import _distance_to_nearest_coin
    t_bfs = bench("BFS to nearest coin (shaping)", lambda: _distance_to_nearest_coin(state), 400)

    print("\nPER GRADIENT STEP (paid once every DQN_TRAIN_EVERY steps)")
    buf = ReplayBuffer(5000)
    for _ in range(2000):
        buf.push(feats, 0, feats, 0.1, False)

    def train_step():
        states, actions, next_states, rewards, dones = buf.sample(32)
        q = net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best = net(next_states).argmax(dim=1, keepdim=True)
            nq = target(next_states).gather(1, best).squeeze(1)
            tgt = rewards + 0.95 * nq * (1 - dones)
        loss = loss_fn(q, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    t_train = bench("full Double DQN update, batch 32", train_step, 60)

    every = 4
    per_step = t_feat + t_fwd + t_bfs + t_train / every
    print("\nMODELLED COST OF ONE 400-STEP ROUND")
    print("-" * 46)
    for label, value in [
        ("state_to_features", t_feat * 400),
        ("network forward (act)", t_fwd * 400),
        ("BFS shaping", t_bfs * 400),
        (f"gradient steps (1 per {every})", t_train * 400 / every),
    ]:
        print(f"{label:<38}{value:8.2f} s   {100 * value / (per_step * 400):5.1f}%")
    print("-" * 46)
    print(f"{'total agent cost per round':<38}{per_step * 400:8.2f} s")
    print("\n(Anything left over in a real round is the game engine itself.)")


if __name__ == "__main__":
    main()
