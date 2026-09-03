#!/usr/bin/env python3
"""Train in chunks and evaluate the FROZEN model after each chunk.

Why this exists. The per-round numbers written by train.py describe an agent
whose weights are changing every few steps. That flatters it badly: on
coin-heaven the online agent logged 45 coins per round while the exact same
saved weights, frozen, scored 7. The weight drift was breaking the action loops
that a frozen policy falls into. Any training curve taken from the online agent
is measuring something that will never be submitted.

So: train N rounds, stop, load the saved weights with training off, measure,
write one row, repeat. The resulting CSV is an honest learning curve and is what
belongs in the report.

    python tools/train_eval_curve.py --chunks 12 --rounds-per-chunk 250 \
        --eval-rounds 40 --scenario coin-heaven
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from evaluate import run as run_stats, mean_ci  # noqa: E402

AGENT = "dqn_agent"
CURVE = REPO / "agent_code" / AGENT / "eval_curve.csv"


def train_chunk(rounds, scenario, resume, env_extra):
    env = {**os.environ, **env_extra, "DQN_RESUME": "1" if resume else "0"}
    cmd = [
        sys.executable, "main.py", "play", "--no-gui",
        "--agents", AGENT, "--train", "1",
        "--scenario", scenario, "--n-rounds", str(rounds),
    ]
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, env=env)
    if result.returncode != 0:
        sys.stderr.write(result.stderr.decode()[-3000:])
        raise SystemExit(f"training chunk failed ({result.returncode})")


def evaluate(rounds, scenario, env_extra):
    stats = run_stats([AGENT], scenario, rounds, seed=None, extra_env=env_extra)
    rounds_data = stats["by_round"].values()
    coins = [r.get("coins", 0) for r in rounds_data]
    steps = [r.get("steps", 0) for r in rounds_data]
    invalid = stats["by_agent"].get(AGENT, {}).get("invalid", 0)
    mean, ci = mean_ci(coins)
    return mean, ci, sum(steps) / max(1, len(steps)), invalid / max(1, len(steps))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", type=int, default=12)
    p.add_argument("--rounds-per-chunk", type=int, default=250)
    p.add_argument("--eval-rounds", type=int, default=40)
    p.add_argument("--scenario", default="coin-heaven")
    p.add_argument("--fresh", action="store_true", help="ignore any existing weights")
    args = p.parse_args()

    # Passed to both training and evaluation so the two never disagree about
    # the action space or the view.
    env_extra = {k: v for k, v in os.environ.items() if k.startswith("DQN_")}
    env_extra.pop("DQN_RESUME", None)

    with open(CURVE, "w", newline="") as fh:
        csv.writer(fh).writerow(
            ["rounds_trained", "frozen_coins", "ci95", "mean_steps", "invalid_per_round", "minutes"]
        )

    start = time.time()
    trained = 0
    for chunk in range(args.chunks):
        resume = not (chunk == 0 and args.fresh)
        train_chunk(args.rounds_per_chunk, args.scenario, resume, env_extra)
        trained += args.rounds_per_chunk

        coins, ci, steps, invalid = evaluate(args.eval_rounds, args.scenario, env_extra)
        minutes = (time.time() - start) / 60

        with open(CURVE, "a", newline="") as fh:
            csv.writer(fh).writerow([
                trained, round(coins, 2), round(ci, 2), round(steps, 1),
                round(invalid, 2), round(minutes, 1),
            ])
        print(f"{trained:>6} rounds | frozen coins {coins:5.2f} +/- {ci:4.2f} "
              f"| steps {steps:5.1f} | invalid/round {invalid:5.2f} | {minutes:5.1f} min",
              flush=True)


if __name__ == "__main__":
    main()
