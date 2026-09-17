#!/usr/bin/env python3
"""Check the 8-fold dihedral augmentation against the real feature pipeline.

Worth a dedicated test rather than trusting the algebra. If the action
permutation is wrong, the network is taught that walking left leads where
walking up leads. Nothing in the loss curve or the reward would look wrong; the
agent would just quietly stop improving, and we have already lost several days
of this project to exactly that class of silent error.

    python tools/test_augment.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("DQN_VIEW", "ego:13")

import settings as s  # noqa: E402
from agent_code.dqn_agent import callbacks as cb  # noqa: E402
from agent_code.dqn_agent.model import (  # noqa: E402
    ACTIONS, N_ACTIONS, TRANSFORMS, action_permutation, transform_board,
    ReplayBuffer,
)

DIRECTION = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}
failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


def empty_board():
    """A board with no crates, so only the symmetry is under test."""
    field = np.zeros((s.COLS, s.ROWS), dtype=int)
    field[:1, :] = field[-1:, :] = field[:, :1] = field[:, -1:] = -1
    for x in range(s.COLS):
        for y in range(s.ROWS):
            if (x + 1) * (y + 1) % 2 == 1:
                field[x, y] = -1
    return field


def state(field, pos, coins=(), others=(), bombs=()):
    return {
        "round": 1, "step": 1, "field": field,
        "self": ("dqn", 0, True, tuple(pos)),
        "others": [("o", 0, True, tuple(p)) for p in others],
        "bombs": list(bombs), "coins": [tuple(c) for c in coins],
        "user_input": None,
        "explosion_map": np.zeros(field.shape),
    }


def rotate_point(p, k, flip, n):
    """Apply the same transform to a coordinate that transform_board applies
    to the array: flip x first, then rot90 k times."""
    x, y = p
    if flip:
        x = n - 1 - x
    for _ in range(k):
        x, y = n - 1 - y, x
    return x, y


print("\n1. The permutation is a group action on the six actions")
for k, flip in TRANSFORMS:
    perm = action_permutation(k, flip)
    check(f"k={k} flip={flip} is a permutation",
          sorted(perm.tolist()) == list(range(N_ACTIONS)), perm.tolist())
    check(f"k={k} flip={flip} fixes WAIT and BOMB",
          perm[4] == 4 and perm[5] == 5, perm.tolist())
check("rot90 alone cycles UP->RIGHT->DOWN->LEFT",
      action_permutation(1, False)[:4].tolist() == [1, 2, 3, 0])
check("flip alone swaps RIGHT and LEFT only",
      action_permutation(0, True)[:4].tolist() == [0, 3, 2, 1])
check("four rotations are the identity",
      action_permutation(4 % 4, False).tolist() == list(range(N_ACTIONS)))

print("\n2. A coin in direction D lands in direction perm[D] (egocentric view)")
field = empty_board()
centre = (s.COLS // 2, s.ROWS // 2)
for k, flip in TRANSFORMS:
    perm = action_permutation(k, flip)
    for name, (dx, dy) in DIRECTION.items():
        gs = state(field, centre, coins=[(centre[0] + dx, centre[1] + dy)])
        feats = torch.from_numpy(cb.state_to_features(gs)).unsqueeze(0)
        moved = transform_board(feats, k, flip)[0]

        size = moved.shape[-1]
        c = size // 2
        # The agent sits at the centre of the egocentric window, and the centre
        # is a fixed point of all eight symmetries of an odd-sized square.
        self_at = (moved[2] > 0).nonzero()
        coin_at = (moved[4] > 0).nonzero()
        expect_name = ACTIONS[perm[ACTIONS.index(name)]]
        ex, ey = DIRECTION[expect_name]

        ok = (self_at.shape[0] == 1 and tuple(self_at[0].tolist()) == (c, c)
              and coin_at.shape[0] == 1
              and tuple(coin_at[0].tolist()) == (c + ex, c + ey))
        check(f"k={k} flip={str(flip):<5} {name:<5} -> {expect_name}", ok,
              "" if ok else f"coin at {coin_at.tolist()}, expected {(c + ex, c + ey)}")

print("\n3. Transforming the world equals transforming the features")
rng = np.random.default_rng(7)
for k, flip in TRANSFORMS:
    f = empty_board()
    crates = rng.random(f.shape) < 0.4
    f[(f == 0) & crates] = 1
    pos = (3, 5)
    f[pos] = 0
    coin = (7, 11)
    f[coin] = 0
    gs = state(f, pos, coins=[coin])

    n = s.COLS
    f2 = np.rot90(np.flip(f, 0) if flip else f, k, axes=(0, 1)).copy()
    gs2 = state(f2, rotate_point(pos, k, flip, n),
                coins=[rotate_point(coin, k, flip, n)])

    direct = torch.from_numpy(cb.state_to_features(gs2, view="global"))
    via = transform_board(
        torch.from_numpy(cb.state_to_features(gs, view="global")), k, flip)
    check(f"k={k} flip={flip} world vs feature transform agree",
          torch.equal(direct, via),
          f"max diff {(direct - via).abs().max().item():.3f}")

print("\n4. The replay buffer's augmentation preserves the batch contract")
buf = ReplayBuffer(500, augment=True)
gs = state(empty_board(), centre, coins=[(centre[0] + 1, centre[1])])
feats = cb.state_to_features(gs)
legal = cb.legal_actions(gs).numpy()
for _ in range(200):
    buf.push(feats, ACTIONS.index("RIGHT"), feats, 1.0, False, legal)
st, ac, nx, rw, dn, nl = buf.sample(128)
check("shapes unchanged", st.shape[1:] == torch.from_numpy(feats).shape
      and ac.shape == (128,) and nl.shape == (128, N_ACTIONS))
check("actions stay in range", bool(((ac >= 0) & (ac < N_ACTIONS)).all()))
check("values stay in [0, 1]", bool((st >= 0).all() and (st <= 1).all()))
check("more than one orientation appears in a batch",
      len(set(ac.tolist())) > 1, f"actions seen: {sorted(set(ac.tolist()))}")
# WAIT is always legal, and WAIT is invariant, so it must survive every
# permutation of the mask.
check("WAIT stays legal after permuting the mask",
      bool(nl[:, ACTIONS.index("WAIT")].all()))
plain = ReplayBuffer(500, augment=False)
for _ in range(64):
    plain.push(feats, ACTIONS.index("RIGHT"), feats, 1.0, False, legal)
pst, pac, _, _, _, pnl = plain.sample(32)
check("augmentation off leaves the board untouched",
      torch.allclose(pst[0], torch.from_numpy(feats), atol=1 / 255))
check("augmentation off leaves the action untouched",
      bool((pac == ACTIONS.index("RIGHT")).all()))
check("augmentation off leaves the legal mask untouched",
      bool((pnl == torch.from_numpy(legal)).all()))

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("all augmentation checks passed")
