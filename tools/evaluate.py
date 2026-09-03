#!/usr/bin/env python3
"""Measure an agent against a fixed set of rounds and report gate metrics.

Starting point for the team's evaluation harness. Wraps main.py rather than
reimplementing the game loop, so it can never drift from the real environment.

    python tools/evaluate.py --agents dqn_agent --scenario coin-heaven --n-rounds 200

Always runs with training off, which is also how the tournament and the
official submission test run your agent.
"""

import argparse
import json
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def mean_ci(values):
    """Mean and half-width of the 95% confidence interval."""
    if not values:
        return 0.0, 0.0
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    stderr = statistics.stdev(values) / math.sqrt(len(values))
    return mean, 1.96 * stderr


def run(agents, scenario, n_rounds, seed, extra_env=None):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        stats_path = tmp.name

    cmd = [
        sys.executable, "main.py", "play", "--no-gui",
        "--agents", *agents,
        "--scenario", scenario,
        "--n-rounds", str(n_rounds),
        "--save-stats", stats_path,
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]

    env = None
    if extra_env:
        import os
        env = {**os.environ, **extra_env}

    result = subprocess.run(cmd, cwd=REPO, capture_output=True, env=env)
    if result.returncode != 0:
        sys.stderr.write(result.stderr.decode()[-3000:])
        raise SystemExit(f"main.py exited with {result.returncode}")

    with open(stats_path) as fh:
        return json.load(fh)


def report(stats, agent, n_rounds):
    by_agent = stats["by_agent"].get(agent, {})
    rounds = stats["by_round"]

    coins = [r.get("coins", 0) for r in rounds.values()]
    steps = [r.get("steps", 0) for r in rounds.values()]
    suicides = sum(r.get("suicides", 0) for r in rounds.values())
    kills = sum(r.get("kills", 0) for r in rounds.values())

    # Only present in by_agent when nonzero: the framework uses a defaultdict.
    invalid = by_agent.get("invalid", 0)
    score = by_agent.get("score", 0)
    seconds = by_agent.get("time", 0.0)
    agent_steps = by_agent.get("steps", 0) or 1

    coin_mean, coin_ci = mean_ci(coins)
    step_mean, step_ci = mean_ci(steps)

    print(f"\nAgent: {agent}   rounds: {len(rounds)}")
    print("-" * 52)
    print(f"{'mean score':<28}{score / max(1, len(rounds)):>10.2f}")
    print(f"{'mean coins / round':<28}{coin_mean:>10.2f}  +/- {coin_ci:.2f}")
    print(f"{'mean steps / round':<28}{step_mean:>10.2f}  +/- {step_ci:.2f}")
    print(f"{'invalid actions / round':<28}{invalid / max(1, len(rounds)):>10.2f}")
    print(f"{'suicide rate':<28}{suicides / max(1, len(rounds)):>10.2%}")
    print(f"{'kills / round':<28}{kills / max(1, len(rounds)):>10.2f}")
    print(f"{'mean decision time (ms)':<28}{1000 * seconds / agent_steps:>10.3f}")
    return {"coins": coin_mean, "coins_ci": coin_ci, "invalid": invalid / max(1, len(rounds))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", nargs="+", required=True)
    parser.add_argument("--scenario", default="classic")
    parser.add_argument("--n-rounds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gate", choices=["1"], help="also check a curriculum gate")
    args = parser.parse_args()

    stats = run(args.agents, args.scenario, args.n_rounds, args.seed)
    summary = report(stats, args.agents[0], args.n_rounds)

    if args.gate == "1":
        print("\nGate 1 (coin-heaven: 45 of 50 coins, zero invalid actions)")
        print("-" * 52)
        coins_ok = summary["coins"] >= 45
        invalid_ok = summary["invalid"] == 0
        print(f"  coins   {summary['coins']:.2f} / 50   {'PASS' if coins_ok else 'FAIL'}")
        print(f"  invalid {summary['invalid']:.2f}       {'PASS' if invalid_ok else 'FAIL'}")
        print(f"\n  {'GATE PASSED' if coins_ok and invalid_ok else 'GATE NOT PASSED'}")


if __name__ == "__main__":
    main()
