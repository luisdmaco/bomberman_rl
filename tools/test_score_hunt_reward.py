#!/usr/bin/env python3
"""Focused checks for the targeted score_hunt bomb reward."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import events as e  # noqa: E402
from agent_code.dqn_agent import train  # noqa: E402


def open_field(size=9):
    field = np.zeros((size, size), dtype=int)
    field[0, :] = field[-1, :] = -1
    field[:, 0] = field[:, -1] = -1
    return field


def state(field, me, others=()):
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("dqn", 0, True, tuple(me)),
        "others": [(f"o{i}", 0, True, tuple(pos)) for i, pos in enumerate(others)],
        "bombs": [],
        "coins": [],
        "user_input": None,
        "explosion_map": np.zeros_like(field),
    }


def bomb_events(game_state):
    return train._bomb_events(
        None, game_state, "BOMB", None, [e.BOMB_DROPPED]
    )


def reward(profile, events):
    agent = SimpleNamespace(rewards=profile, cfg={"shaping": 0})
    return train._reward(agent, events, 0.0, 0.0)


def check(label, condition):
    if not condition:
        raise AssertionError(label)
    print(f"  ok  {label}")


def main():
    target = state(open_field(), (4, 4), [(4, 6)])
    target_events = bomb_events(target)
    check("escapable opponent bomb is useful",
          train.GOOD_BOMB in target_events)
    check("escapable opponent bomb gets the targeted event",
          train.OPPONENT_THREATENED in target_events)
    check("targeted opponent bomb is not labelled suicidal",
          train.SUICIDAL_BOMB not in target_events)

    crate_field = open_field()
    crate_field[4, 6] = 1
    crate_events = bomb_events(state(crate_field, (4, 4)))
    check("crate-only bomb remains useful",
          train.GOOD_BOMB in crate_events)
    check("crate-only bomb gets no opponent event",
          train.OPPONENT_THREATENED not in crate_events)

    blocked_field = open_field()
    blocked_field[4, 5] = -1
    blocked_events = bomb_events(state(blocked_field, (4, 4), [(4, 6)]))
    check("stone wall blocks the opponent blast line",
          train.OPPONENT_THREATENED not in blocked_events)

    trapped_field = np.full((7, 7), -1, dtype=int)
    trapped_field[1, 1] = trapped_field[2, 1] = trapped_field[3, 1] = 0
    trapped_events = bomb_events(state(trapped_field, (1, 1), [(3, 1)]))
    check("inescapable aimed bomb is labelled suicidal",
          train.SUICIDAL_BOMB in trapped_events)
    check("inescapable aimed bomb gets no hunting bonus",
          train.OPPONENT_THREATENED not in trapped_events)

    legacy = reward(train.SCORE_REWARDS, [train.OPPONENT_THREATENED])
    hunt = reward(train.SCORE_HUNT_REWARDS, [train.OPPONENT_THREATENED])
    check("historical score profile remains neutral", abs(legacy + 0.02) < 1e-9)
    check("score_hunt pays exactly +0.5 before step cost", abs(hunt - 0.48) < 1e-9)
    check("crate-only GOOD_BOMB stays neutral in score_hunt",
          abs(reward(train.SCORE_HUNT_REWARDS, [train.GOOD_BOMB]) + 0.02) < 1e-9)

    print("all score_hunt reward checks passed")


if __name__ == "__main__":
    main()
