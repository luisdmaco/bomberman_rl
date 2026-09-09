#!/usr/bin/env python3
"""Train once, continuously, then build an honest learning curve from checkpoints.

This replaces tools/train_eval_curve.py, which was wrong. That script trained in
chunks by re-invoking main.py, and each invocation is a fresh process: epsilon
restarted at 1.0 and never annealed below ~0.70, the replay buffer was wiped,
and the Adam optimiser state was thrown away. Only the network weights carried
over. The result was 1500 rounds of near-random experience and a flat curve.

The shape here is: ONE training process, so epsilon, replay buffer and optimiser
state are all continuous. It drops numbered checkpoints as it goes. Afterwards
each checkpoint is replayed with training switched off, which is the only
measurement that describes the agent you would actually submit.

    python tools/train_curve.py --rounds 3000 --checkpoint-every 250 --eval-rounds 40

Add --eval-only to rebuild the curve from checkpoints already on disk.
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
AGENT_DIR = REPO / "agent_code" / AGENT
CHECKPOINTS = AGENT_DIR / "checkpoints"
CURVE = AGENT_DIR / "eval_curve.csv"


def dqn_env():
    """Every DQN_* variable the caller set, minus the ones we control."""
    env = {k: v for k, v in os.environ.items() if k.startswith("DQN_")}
    for controlled in ("DQN_RESUME", "DQN_MODEL_FILE", "DQN_CHECKPOINT_EVERY"):
        env.pop(controlled, None)
    return env


def archive_checkpoints():
    """Move any existing checkpoints aside before a new training run.

    Without this, a shorter run leaves the previous run's higher-numbered
    checkpoints in place, the curve is then built from `glob("dqn-r*.pt")`, and
    the tail of eval_curve.csv silently reports a *different run's* models
    measured on the current scenario. That happened on 2026-09-09: a 4000-round
    classic run produced a curve with rows out to 6000, and the last eight rows
    were stale crate-light models. It also means one run can overwrite another
    run's weights, which cost us run 1's checkpoints the same day.

    Returns the archive directory, or None if there was nothing to move.
    """
    existing = sorted(CHECKPOINTS.glob("dqn-r*.pt"))
    if not existing:
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = CHECKPOINTS / f"archive-{stamp}"
    dest.mkdir(parents=True)
    for path in existing:
        path.rename(dest / path.name)
    print(f"Moved {len(existing)} existing checkpoints to {dest.name}/", flush=True)
    return dest


def train(rounds, checkpoint_every, scenario, resume, env_extra):
    env = {
        **os.environ, **env_extra,
        "DQN_CHECKPOINT_EVERY": str(checkpoint_every),
        "DQN_RESUME": "1" if resume else "0",
    }
    cmd = [
        sys.executable, "main.py", "play", "--no-gui",
        "--agents", AGENT, "--train", "1",
        "--scenario", scenario, "--n-rounds", str(rounds),
    ]
    print(f"Training {rounds} rounds in one continuous run...", flush=True)
    started = time.time()
    result = subprocess.run(cmd, cwd=REPO, env=env)
    if result.returncode != 0:
        raise SystemExit(f"training failed ({result.returncode})")
    print(f"Training done in {(time.time() - started) / 60:.1f} min", flush=True)


def evaluate_checkpoint(path, rounds, scenario, env_extra):
    env = {**env_extra, "DQN_MODEL_FILE": str(path)}
    stats = run_stats([AGENT], scenario, rounds, seed=None, extra_env=env)
    per_round = stats["by_round"].values()
    coins = [r.get("coins", 0) for r in per_round]
    steps = [r.get("steps", 0) for r in per_round]
    invalid = stats["by_agent"].get(AGENT, {}).get("invalid", 0)
    mean, ci = mean_ci(coins)
    return mean, ci, sum(steps) / max(1, len(steps)), invalid / max(1, len(steps))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=3000)
    p.add_argument("--checkpoint-every", type=int, default=250)
    p.add_argument("--eval-rounds", type=int, default=40)
    p.add_argument("--scenario", default="coin-heaven")
    p.add_argument("--resume", action="store_true", help="continue from existing weights")
    p.add_argument("--eval-only", action="store_true", help="skip training, just rebuild the curve")
    args = p.parse_args()

    env_extra = dqn_env()

    if not args.eval_only:
        archive_checkpoints()
        train(args.rounds, args.checkpoint_every, args.scenario, args.resume, env_extra)

    checkpoints = sorted(CHECKPOINTS.glob("dqn-r*.pt"))
    # Belt and braces: even with the archive step, never plot a checkpoint from
    # beyond this run's horizon. A row labelled r6000 on a 4000-round run is a
    # different run's model and the curve must not imply otherwise.
    checkpoints = [c for c in checkpoints
                   if int(c.stem.split("-r")[1]) <= args.rounds]
    if not checkpoints:
        raise SystemExit(f"no checkpoints in {CHECKPOINTS}")

    with open(CURVE, "w", newline="") as fh:
        csv.writer(fh).writerow(
            ["rounds_trained", "frozen_coins", "ci95", "mean_steps", "invalid_per_round"]
        )

    print(f"\nEvaluating {len(checkpoints)} checkpoints, {args.eval_rounds} rounds each, "
          f"training OFF:\n", flush=True)
    for path in checkpoints:
        trained = int(path.stem.split("-r")[1])
        coins, ci, steps, invalid = evaluate_checkpoint(
            path, args.eval_rounds, args.scenario, env_extra
        )
        with open(CURVE, "a", newline="") as fh:
            csv.writer(fh).writerow([
                trained, round(coins, 2), round(ci, 2), round(steps, 1), round(invalid, 2),
            ])
        print(f"{trained:>6} rounds | frozen coins {coins:5.2f} +/- {ci:4.2f} "
              f"| steps {steps:5.1f} | invalid/round {invalid:5.2f}", flush=True)

    print(f"\nCurve written to {CURVE.relative_to(REPO)}")
    print("Reference: random walk ~18 coins, rule_based_agent 50.0, gate 1 needs 45.")


if __name__ == "__main__":
    main()
