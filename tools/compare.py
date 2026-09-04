#!/usr/bin/env python3
"""Evaluate every sweep configuration frozen, and put them in one table.

Each config's checkpoints are replayed with training switched off, which is the
only measurement that describes the agent that would be submitted. Configs are
evaluated in parallel, one core each, the same way they were trained.

    python tools/compare.py --eval-rounds 40 --jobs 6

Writes eval_curve.csv into each agent directory and a combined
experiments/sweep_results.csv, which is tracked by git so the report can be
built from it months later.

Critically, each config is evaluated with the SAME environment overrides it was
trained with. Evaluating dqn_global_view without DQN_VIEW=global would build a
network of the wrong shape and fail to load its own weights.
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from evaluate import run as run_stats, mean_ci  # noqa: E402
from sweep import RUNS, COMMON  # noqa: E402

RESULTS = REPO / "experiments" / "sweep_results.csv"


def env_for(name):
    env = {k: v for k, v in COMMON.items()}
    env.update(RUNS.get(name, {}))
    return env


def evaluate_one(path, agent, rounds, scenario, env):
    stats = run_stats([agent], scenario, rounds, seed=None,
                      extra_env={**env, "DQN_MODEL_FILE": str(path)})
    per_round = stats["by_round"].values()
    coins = [r.get("coins", 0) for r in per_round]
    steps = [r.get("steps", 0) for r in per_round]
    invalid = stats["by_agent"].get(agent, {}).get("invalid", 0)
    mean, ci = mean_ci(coins)
    return mean, ci, sum(steps) / max(1, len(steps)), invalid / max(1, len(steps))


def evaluate_config(job):
    name, rounds, scenario = job
    agent = f"dqn_{name}"
    agent_dir = REPO / "agent_code" / agent
    checkpoints = sorted((agent_dir / "checkpoints").glob("dqn-r*.pt"))
    if not checkpoints:
        return name, [], "no checkpoints"

    env = env_for(name)
    curve = []
    for path in checkpoints:
        trained = int(path.stem.split("-r")[1])
        try:
            coins, ci, steps, invalid = evaluate_one(path, agent, rounds, scenario, env)
        except SystemExit as exc:
            return name, curve, str(exc)
        curve.append((trained, coins, ci, steps, invalid))

    with open(agent_dir / "eval_curve.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rounds_trained", "frozen_coins", "ci95", "mean_steps", "invalid_per_round"])
        for row in curve:
            w.writerow([row[0], round(row[1], 2), round(row[2], 2),
                        round(row[3], 1), round(row[4], 2)])
    return name, curve, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-rounds", type=int, default=40)
    p.add_argument("--scenario", default="coin-heaven")
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--only", nargs="*")
    args = p.parse_args()

    names = args.only or [
        d.name.replace("dqn_", "") for d in sorted((REPO / "agent_code").glob("dqn_*"))
        if d.name != "dqn_agent" and (d / "checkpoints").is_dir()
    ]
    if not names:
        raise SystemExit("no sweep configs with checkpoints found")

    print(f"Evaluating {len(names)} configs, {args.eval_rounds} rounds per checkpoint, "
          f"training OFF, {args.jobs} at a time.\n")
    started = time.time()

    results = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for name, curve, error in pool.map(
            evaluate_config, [(n, args.eval_rounds, args.scenario) for n in names]
        ):
            if error:
                print(f"{name:<14} FAILED: {error}")
                continue
            results[name] = curve
            best = max(curve, key=lambda r: r[1])
            print(f"{name:<14} done, best {best[1]:5.2f} coins at {best[0]} rounds", flush=True)

    print(f"\nEvaluated in {(time.time() - started) / 60:.1f} min\n")

    RESULTS.parent.mkdir(exist_ok=True)
    with open(RESULTS, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "rounds_trained", "frozen_coins", "ci95",
                    "mean_steps", "invalid_per_round"])
        for name, curve in results.items():
            for row in curve:
                w.writerow([name, row[0], round(row[1], 2), round(row[2], 2),
                            round(row[3], 1), round(row[4], 2)])

    print(f"{'config':<14}{'final':>8}{'+/-':>7}{'best':>8}{'@rounds':>9}"
          f"{'steps':>8}{'invalid':>9}")
    print("-" * 63)
    for name, curve in sorted(results.items(), key=lambda kv: -max(r[1] for r in kv[1])):
        final = curve[-1]
        best = max(curve, key=lambda r: r[1])
        print(f"{name:<14}{final[1]:>8.2f}{final[2]:>7.2f}{best[1]:>8.2f}{best[0]:>9}"
              f"{final[3]:>8.0f}{final[4]:>9.2f}")

    print("\nReference: random walk ~18 coins, rule_based_agent 50.0, gate 1 needs 45.")
    print("mean_steps of 400 means the board was never cleared.")
    print(f"\nCombined results: {RESULTS.relative_to(REPO)}")


if __name__ == "__main__":
    main()
