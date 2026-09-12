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

From task 3 on, pass --opponents. They join the training game AND every frozen
evaluation, and the curve then carries the task-3 numbers (suicide rate,
kill-in-round, score margin) instead of coins alone. Coins on their own stopped
meaning anything the moment somebody else started collecting them.

    python tools/train_curve.py --rounds 4000 --scenario classic \
        --opponents coin_collector_agent --resume

Add --eval-only to rebuild the curve from checkpoints already on disk.
"""

import argparse
import csv
import json
import os
import statistics
import tempfile
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


def train(rounds, checkpoint_every, scenario, resume, env_extra, opponents=()):
    env = {
        **os.environ, **env_extra,
        "DQN_CHECKPOINT_EVERY": str(checkpoint_every),
        "DQN_RESUME": "1" if resume else "0",
    }
    cmd = [
        sys.executable, "main.py", "play", "--no-gui",
        "--agents", AGENT, *opponents, "--train", "1",
        "--scenario", scenario, "--n-rounds", str(rounds),
    ]
    against = f" against {', '.join(opponents)}" if opponents else ""
    print(f"Training {rounds} rounds{against} in one continuous run...", flush=True)
    started = time.time()
    result = subprocess.run(cmd, cwd=REPO, env=env)
    if result.returncode != 0:
        raise SystemExit(f"training failed ({result.returncode})")
    print(f"Training done in {(time.time() - started) / 60:.1f} min", flush=True)


SOLO_COLUMNS = ["rounds_trained", "frozen_coins", "ci95", "mean_steps", "invalid_per_round"]
VERSUS_COLUMNS = ["rounds_trained", "frozen_coins", "ci95", "mean_steps",
                  "invalid_per_round", "suicide_rate", "kill_in_round",
                  "score_margin", "margin_ci95"]


def evaluate_checkpoint(path, rounds, scenario, env_extra):
    """Solo. Coins are the whole story only while nobody else is on the board."""
    env = {**env_extra, "DQN_MODEL_FILE": str(path)}
    stats = run_stats([AGENT], scenario, rounds, seed=None, extra_env=env)
    per_round = stats["by_round"].values()
    coins = [r.get("coins", 0) for r in per_round]
    steps = [r.get("steps", 0) for r in per_round]
    invalid = stats["by_agent"].get(AGENT, {}).get("invalid", 0)
    mean, ci = mean_ci(coins)
    return mean, ci, sum(steps) / max(1, len(steps)), invalid / max(1, len(steps))


def evaluate_checkpoint_versus(path, rounds, scenario, opponents, env_extra):
    """With opponents, in a SEPARATE PROCESS. That is not optional.

    `callbacks.MODEL_FILE` is a module-level constant built when the module is
    first imported, and the agent module is cached in sys.modules after that.
    A caller that loops over checkpoints in one process therefore measures the
    FIRST checkpoint sixteen times and draws a flat curve. Shelling out to
    evaluate3.py gives every checkpoint a fresh interpreter.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out = tmp.name
    env = {**os.environ, **env_extra, "DQN_MODEL_FILE": str(path)}
    cmd = [sys.executable, "tools/evaluate3.py",
           "--agents", AGENT, *opponents,
           "--scenario", scenario, "--n-rounds", str(rounds), "--json", out]
    result = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr.decode()[-3000:])
        raise SystemExit(f"evaluate3.py exited with {result.returncode}")
    rows = json.load(open(out))

    n = len(rows)
    # The framework renames duplicates: three rule_based_agents come back as
    # rule_based_agent_0/_1/_2, so the keys in the rows are not the names that
    # were asked for. Always read the keys the results actually carry.
    opp_keys = [k for k in rows[0] if k not in (AGENT, "_steps")]
    mine = [r[AGENT] for r in rows]
    coins, ci = mean_ci([m["coins"] for m in mine])
    steps = sum(r["_steps"] for r in rows) / max(1, n)
    invalid = sum(m["invalid"] for m in mine) / max(1, n)
    suicide = sum(m["suicides"] for m in mine) / max(1, n)
    killed = sum(1 for m in mine if m["kills"] > 0) / max(1, n)
    # Gate 4 is "mean score strictly above rule_based_agent's", so the comparator
    # is a typical opponent, not the luckiest one in the round. With three of
    # them, max() reads about 2.5 points harsher than the gate does.
    margin, margin_ci = mean_ci(
        [r[AGENT]["score"] - statistics.fmean([r[o]["score"] for o in opp_keys])
         for r in rows])
    return dict(coins=coins, ci=ci, steps=steps, invalid=invalid,
                suicide=suicide, killed=killed, margin=margin, margin_ci=margin_ci)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=3000)
    p.add_argument("--checkpoint-every", type=int, default=250)
    p.add_argument("--eval-rounds", type=int, default=40)
    p.add_argument("--scenario", default="coin-heaven")
    p.add_argument("--opponents", nargs="*", default=[],
                   help="agents to train and evaluate against, e.g. coin_collector_agent")
    p.add_argument("--resume", action="store_true", help="continue from existing weights")
    p.add_argument("--eval-only", action="store_true", help="skip training, just rebuild the curve")
    args = p.parse_args()

    env_extra = dqn_env()

    if not args.eval_only:
        archive_checkpoints()
        train(args.rounds, args.checkpoint_every, args.scenario, args.resume,
              env_extra, args.opponents)

    checkpoints = sorted(CHECKPOINTS.glob("dqn-r*.pt"))
    # Belt and braces: even with the archive step, never plot a checkpoint from
    # beyond this run's horizon. A row labelled r6000 on a 4000-round run is a
    # different run's model and the curve must not imply otherwise.
    checkpoints = [c for c in checkpoints
                   if int(c.stem.split("-r")[1]) <= args.rounds]
    if not checkpoints:
        raise SystemExit(f"no checkpoints in {CHECKPOINTS}")

    versus = bool(args.opponents)
    with open(CURVE, "w", newline="") as fh:
        csv.writer(fh).writerow(VERSUS_COLUMNS if versus else SOLO_COLUMNS)

    against = f" against {', '.join(args.opponents)}" if versus else ""
    print(f"\nEvaluating {len(checkpoints)} checkpoints, {args.eval_rounds} rounds each"
          f"{against}, training OFF:\n", flush=True)
    for path in checkpoints:
        trained = int(path.stem.split("-r")[1])
        if versus:
            m = evaluate_checkpoint_versus(path, args.eval_rounds, args.scenario,
                                           args.opponents, env_extra)
            row = [trained, round(m["coins"], 2), round(m["ci"], 2), round(m["steps"], 1),
                   round(m["invalid"], 2), round(m["suicide"], 4), round(m["killed"], 4),
                   round(m["margin"], 2), round(m["margin_ci"], 2)]
            print(f"{trained:>6} rounds | coins {m['coins']:5.2f} +/- {m['ci']:4.2f} "
                  f"| suicide {m['suicide']:6.2%} | kill-in-round {m['killed']:6.2%} "
                  f"| margin {m['margin']:+6.2f} +/- {m['margin_ci']:4.2f}", flush=True)
        else:
            coins, ci, steps, invalid = evaluate_checkpoint(
                path, args.eval_rounds, args.scenario, env_extra)
            row = [trained, round(coins, 2), round(ci, 2), round(steps, 1), round(invalid, 2)]
            print(f"{trained:>6} rounds | frozen coins {coins:5.2f} +/- {ci:4.2f} "
                  f"| steps {steps:5.1f} | invalid/round {invalid:5.2f}", flush=True)
        with open(CURVE, "a", newline="") as fh:
            csv.writer(fh).writerow(row)

    print(f"\nCurve written to {CURVE.relative_to(REPO)}")
    if versus:
        print("Gate 3: kill peaceful_agent in >= 80% of rounds; positive margin vs "
              "coin_collector_agent.")
        print("Baseline, r3750 frozen: vs coin_collector 13.25% suicide, +0.17 margin; "
              "vs peaceful 69.2% kill-in-round.")
        print("These are 40-round points. Confirm any candidate with tools/evaluate3.py "
              "at 200+ rounds before believing it.")
    else:
        print("Reference: random walk ~18 coins, rule_based_agent 50.0, gate 1 needs 45.")


if __name__ == "__main__":
    main()
