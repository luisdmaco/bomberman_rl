#!/usr/bin/env python3
"""Per-agent, per-round evaluation. Needed from task 3 onward.

`tools/evaluate.py` reads `--save-stats`, and the framework's round statistics
sum `coins`, `kills` and `suicides` **over all agents**. Solo that is the same
thing as our agent's numbers; with an opponent on the board it is not, and
`coin_collector_agent` does drop bombs, so an aggregated kill count cannot tell
us who killed whom. Both task-3 gates need a per-agent, per-round breakdown:

    kills the peaceful agent in >= 80% of ROUNDS   -> needs per round
    positive score margin vs coin_collector        -> needs per agent

So this drives `BombeRLeWorld` itself and reads each agent's `statistics` dict
at the end of every round, before `start_round` clears it. Same world class,
same `do_step`, same stopping rules as `main.py`; only the bookkeeping differs.

Always runs with training off.

    python tools/evaluate3.py --agents dqn_agent peaceful_agent --n-rounds 200
    python tools/evaluate3.py --agents dqn_agent coin_collector_agent --n-rounds 200 --gate 3
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from environment import BombeRLeWorld, WorldArgs  # noqa: E402

KEYS = ["score", "coins", "kills", "suicides", "crates", "bombs", "invalid"]


def mean_ci(values):
    if not values:
        return 0.0, 0.0
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def wilson_upper(successes, n, z=1.96):
    """95% upper bound on a rate. Used the same way as for the suicide gate."""
    if n == 0:
        return 1.0
    p = successes / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre + half) / d


def wilson_lower(successes, n, z=1.96):
    if n == 0:
        return 0.0
    p = successes / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / d


def play(agent_names, scenario, n_rounds, seed):
    args = WorldArgs(
        no_gui=True, fps=15, turn_based=False, update_interval=0.1,
        save_replay=False, replay=None, make_video=False,
        continue_without_training=True, log_dir=str(REPO / "logs"),
        save_stats=False, match_name="evaluate3", seed=seed,
        silence_errors=False, scenario=scenario,
    )
    world = BombeRLeWorld(args, [(name, False) for name in agent_names])
    world.user_input = None

    rounds = []
    for _ in range(n_rounds):
        world.new_round()
        while world.running:
            world.do_step()
        rounds.append({a.name: {k: a.statistics[k] for k in KEYS}
                       for a in world.agents})
        rounds[-1]["_steps"] = world.step
    world.end()
    return rounds


def agent_keys(rounds, requested):
    """The framework renames duplicates: three rule_based_agents come back as
    rule_based_agent_0/_1/_2. Always read the keys the rows actually carry,
    never the names that were asked for."""
    keys = [k for k in rounds[0] if k != "_steps"]
    me = requested[0] if requested[0] in keys else keys[0]
    return me, [k for k in keys if k != me]


def report(rounds, agent_names, gate):
    n = len(rounds)
    me_key, opp_keys = agent_keys(rounds, agent_names)
    agent_names = [me_key, *opp_keys]
    print(f"\n{n} rounds, training off")
    print("-" * 74)
    print(f"{'':<22}" + "".join(f"{name[:16]:>26}" for name in agent_names))

    per = {name: {k: [r[name][k] for r in rounds] for k in KEYS}
           for name in agent_names}

    for k in KEYS:
        row = f"{'mean ' + k + ' / round':<22}"
        for name in agent_names:
            m, ci = mean_ci(per[name][k])
            row += f"{m:>18.2f} +/-{ci:5.2f}"
        print(row)

    steps, steps_ci = mean_ci([r["_steps"] for r in rounds])
    print(f"{'mean steps / round':<22}{steps:>18.2f} +/-{steps_ci:5.2f}")

    me = me_key
    killed = sum(1 for r in rounds if r[me]["kills"] > 0)
    print(f"\n{me}:")
    print(f"  rounds with at least one kill   {killed}/{n} = {killed / n:.1%}"
          f"   [95% {wilson_lower(killed, n):.1%}, {wilson_upper(killed, n):.1%}]")
    suicides = sum(r[me]["suicides"] for r in rounds)
    print(f"  suicide rate                    {suicides / n:.2%}"
          f"   (95% upper {wilson_upper(suicides, n):.2%})")

    margins = {}
    for other in agent_names[1:]:
        diff = [r[me]["score"] - r[other]["score"] for r in rounds]
        m, ci = mean_ci(diff)
        margins[other] = (m, ci)
        print(f"  score margin vs {other[:22]:<22} {m:>6.2f} +/- {ci:.2f}"
              f"   [{m - ci:.2f}, {m + ci:.2f}]")

    if gate == "3":
        print("\nGate 3 (task 3: hunting)")
        print("-" * 74)
        if any(k.startswith("peaceful_agent") for k in agent_names):
            lower = wilson_lower(killed, n)
            ok = lower >= 0.80
            print(f"  kills peaceful_agent in >= 80% of rounds:"
                  f" {killed / n:.1%}, 95% lower bound {lower:.1%}"
                  f"   {'PASS' if ok else 'FAIL'}")
        for other, (m, ci) in margins.items():
            if other.startswith("peaceful_agent"):
                continue
            ok = m - ci > 0
            print(f"  positive score margin vs {other}:"
                  f" {m:+.2f} +/- {ci:.2f}   {'PASS' if ok else 'FAIL'}"
                  f"  (whole interval above zero)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agents", nargs="+", required=True,
                   help="ours first; the rest are opponents")
    p.add_argument("--scenario", default="classic")
    p.add_argument("--n-rounds", type=int, default=200)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--gate", choices=["3"])
    p.add_argument("--json", help="write the per-round rows here instead of printing "
                                  "a table. For tools that shell out to this one, which "
                                  "they must: callbacks.MODEL_FILE is a module-level "
                                  "constant frozen at first import, so a caller that "
                                  "evaluates several checkpoints in one process silently "
                                  "measures the first one every time.")
    args = p.parse_args()

    rounds = play(args.agents, args.scenario, args.n_rounds, args.seed)
    if args.json:
        json.dump(rounds, open(args.json, "w"))
    else:
        report(rounds, args.agents, args.gate)


if __name__ == "__main__":
    main()
