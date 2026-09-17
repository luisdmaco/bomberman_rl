#!/usr/bin/env python3
"""Check that the agent actually perceives opponents. Task-3 groundwork.

Channel 3 of the feature tensor is "others" and has been all-zero for the whole
project, because tasks 1 and 2 have no opponents. Task 3 is the first time it
carries anything, so it gets tested before any result that depends on it is
believed. Same for the two places opponents change the rules: an opponent's
bomb has to register as danger, and an opponent's tile has to count as blocked.

Two halves:
  1. Synthetic states, where the expected answer is known exactly.
  2. A live game against peaceful_agent, driving the real environment, because
     a synthetic dict that the framework never produces proves nothing.

    python tools/test_others.py
"""

import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("DQN_VIEW", "ego:13")

import settings as s  # noqa: E402
from agent_code.dqn_agent import callbacks as cb  # noqa: E402
from agent_code.dqn_agent.model import (  # noqa: E402
    ACTIONS, TRANSFORMS, transform_board,
)

OTHERS = 3          # channel index under test
DANGER = 5
failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


def empty_board():
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


# ---------------------------------------------------------------- synthetic

def test_channel_global():
    print("\n1. channel 3 carries opponents, global view")
    field = empty_board()
    me = (1, 1)

    g = cb.state_to_features(state(field, me), view="global")
    check("no opponents -> channel 3 all zero", g[OTHERS].sum() == 0,
          f"sum={g[OTHERS].sum()}")

    opp = [(1, 3), (5, 1), (9, 9)]
    g = cb.state_to_features(state(field, me, others=opp), view="global")
    check("three opponents -> exactly three lit tiles", g[OTHERS].sum() == 3.0,
          f"sum={g[OTHERS].sum()}")
    check("lit at the right coordinates",
          all(g[OTHERS][x, y] == 1.0 for x, y in opp))
    check("nothing else lit",
          set(zip(*np.nonzero(g[OTHERS]))) == {(x, y) for x, y in opp})
    check("own position stays out of channel 3", g[OTHERS][me] == 0.0)
    check("own position still in channel 2", g[2][me] == 1.0)


def test_channel_ego():
    print("\n2. channel 3 under the ego:13 crop the agent actually uses")
    field = empty_board()
    me = (7, 7)
    size = 13
    c = size // 2

    for name, (dx, dy) in (("UP", (0, -1)), ("RIGHT", (1, 0)),
                           ("DOWN", (0, 1)), ("LEFT", (-1, 0))):
        opp = (me[0] + dx, me[1] + dy)
        f = cb.state_to_features(state(field, me, others=[opp]), view="ego:13")
        check(f"opponent one tile {name} lands at the right offset",
              f[OTHERS][c + dx, c + dy] == 1.0 and f[OTHERS].sum() == 1.0)

    # The crop is the agent's whole world: radius 6 on a 17x17 board.
    near = (me[0] + 6, me[1])
    far = (me[0] + 7, me[1])
    f = cb.state_to_features(state(field, me, others=[near]), view="ego:13")
    check("opponent at the crop edge (6 tiles) is visible", f[OTHERS].sum() == 1.0)
    f = cb.state_to_features(state(field, me, others=[far]), view="ego:13")
    check("opponent 7 tiles away is cropped out (known blind spot)",
          f[OTHERS].sum() == 0.0)

    corner = (1, 1)
    f = cb.state_to_features(state(field, corner, others=[(1, 3)]), view="ego:13")
    check("crop at the board corner still shows a nearby opponent",
          f[OTHERS][c, c + 2] == 1.0 and f[OTHERS].sum() == 1.0)


def test_opponent_bombs():
    print("\n3. an opponent's bomb is danger, not just ours")
    field = empty_board()
    me = (1, 1)
    opp = (1, 5)

    st = state(field, me, others=[opp], bombs=[((1, 3), 2)])
    steps = cb.steps_until_lethal(st)
    check("opponent's bomb makes our tile lethal", steps[me] == 2,
          f"steps_until_lethal at self = {steps[me]}")
    f = cb.state_to_features(st, view="global")
    check("danger channel reflects it",
          abs(f[DANGER][me] - (s.BOMB_TIMER - 2) / s.BOMB_TIMER) < 1e-6,
          f"danger={f[DANGER][me]}")
    check("steps_to_safety sees a way out", cb.steps_to_safety(st) is not None)

    # Same bomb, but ours. The map must not care who owns it.
    mine = state(field, me, bombs=[((1, 3), 2)])
    check("ownership does not change the danger map",
          np.array_equal(cb.steps_until_lethal(st), cb.steps_until_lethal(mine)))


def test_legality_and_escape():
    print("\n4. an opponent's body blocks a tile")
    field = empty_board()
    me = (1, 1)

    free = cb.legal_actions(state(field, me))
    blocked = cb.legal_actions(state(field, me, others=[(1, 2)]))
    check("DOWN is legal with the tile empty", bool(free[ACTIONS.index("DOWN")]))
    check("DOWN is illegal with an opponent standing there",
          not bool(blocked[ACTIONS.index("DOWN")]))
    check("the other directions are unaffected",
          all(free[i] == blocked[i] for i in range(len(ACTIONS))
              if ACTIONS[i] != "DOWN"))

    # A corridor with the only exit occupied: bombing there is suicide, and the
    # play-time survivability mask has to notice.
    st_open = state(field, (1, 1))
    st_walled = state(field, (1, 1), others=[(1, 2), (2, 1)])
    check("escape exists from an open corner",
          cb.escape_exists(st_open, extra_bomb=(1, 1)))
    check("no escape when opponents seal both exits",
          not cb.escape_exists(st_walled, extra_bomb=(1, 1)))


def test_augmentation_with_others():
    print("\n5. the 8-fold augmentation moves opponents with the board")
    field = empty_board()
    me = (7, 7)
    opp = (7, 5)
    n = field.shape[0]

    base = cb.state_to_features(state(field, me, others=[opp]), view="ego:13")
    for k, flip in TRANSFORMS:
        moved = transform_board(
            __import__("torch").from_numpy(base).unsqueeze(0), k, flip)[0].numpy()
        check(f"transform (k={k}, flip={flip}) keeps exactly one opponent",
              moved[OTHERS].sum() == 1.0, f"sum={moved[OTHERS].sum()}")
        check(f"transform (k={k}, flip={flip}) keeps the channels aligned",
              moved[2].sum() == 1.0 and moved[OTHERS].sum() == 1.0)


# --------------------------------------------------------------------- live

def live_game(rounds=3, opponent="peaceful_agent", scenario="classic"):
    print(f"\n6. live game: dqn_agent vs {opponent} on {scenario}, {rounds} rounds")
    from environment import BombeRLeWorld, WorldArgs

    args = WorldArgs(
        no_gui=True, fps=15, turn_based=False, update_interval=0.1,
        save_replay=False, replay=None, make_video=False,
        continue_without_training=True, log_dir=str(REPO / "logs"),
        save_stats=False, match_name="test_others", seed=1,
        silence_errors=False, scenario=scenario,
    )
    world = BombeRLeWorld(args, [("dqn_agent", False), (opponent, False)])
    me = world.agents[0]

    steps = 0
    steps_with_opponent_visible = 0
    steps_opponent_alive = 0
    max_lit = 0
    channel_ever_lit = False

    world.user_input = None

    for _ in range(rounds):
        world.new_round()
        while world.running:
            if not me.dead:
                st = world.get_state_for_agent(me)
                f = cb.state_to_features(st, view=os.environ["DQN_VIEW"])
                lit = int(f[OTHERS].sum())
                steps += 1
                if st["others"]:
                    steps_opponent_alive += 1
                if lit:
                    steps_with_opponent_visible += 1
                    channel_ever_lit = True
                max_lit = max(max_lit, lit)
                # The framework can never report more opponents than exist.
                if len(st["others"]) < lit:
                    check("channel 3 never invents an opponent", False,
                          f"{lit} lit, {len(st['others'])} others")
            world.do_step()
    world.end()

    pct = 100 * steps_with_opponent_visible / max(1, steps_opponent_alive)
    print(f"     {steps} steps, opponent alive for {steps_opponent_alive},"
          f" visible in the 13x13 crop on {steps_with_opponent_visible}"
          f" ({pct:.0f}%), most lit at once {max_lit}")
    check("channel 3 is populated in a real game", channel_ever_lit)
    check("never more lit tiles than opponents", max_lit <= 1)
    check("opponent visible on a meaningful share of steps", pct > 10,
          f"only {pct:.0f}%")


def main():
    print("Opponent perception checks (task 3 groundwork)")
    test_channel_global()
    test_channel_ego()
    test_opponent_bombs()
    test_legality_and_escape()
    test_augmentation_with_others()
    live_game()

    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
