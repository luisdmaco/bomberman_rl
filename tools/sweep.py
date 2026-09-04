#!/usr/bin/env python3
"""Run several training configurations at the same time, one core each.

The project spec forbids multiprocessing in the *submitted agent*, but says
explicitly that "multiprocessing or anything else that comes to your mind to
improve training is perfectly fine". This is that.

A single training run cannot use more than about one core: the game loop is
serial Python and the network is too small for extra threads to help (a second
thread buys under 10%, a second process buys a whole extra run). So the way to
use a 14-core machine is to run the whole ablation at once instead of one
experiment at a time.

Each run gets its own agent directory copy, so checkpoints, logs and curves
never collide.

    python tools/sweep.py --rounds 3000 --jobs 5

Edit RUNS below to change what is compared. The defaults are the ablations the
report needs: shaping on/off, egocentric vs global view, and two learning rates.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE_AGENT = REPO / "agent_code" / "dqn_agent"

# name -> environment overrides for that run
RUNS = {
    "baseline":      {},
    "no_shaping":    {"DQN_SHAPING": "0"},
    "global_view":   {"DQN_VIEW": "global"},
    "lr_2e4":        {"DQN_LR": "0.0002"},
    "batch32":       {"DQN_BATCH": "32", "DQN_TRAIN_EVERY": "4"},
    # Reproduces the reward bug found on 2026-09-05, for the report's ablation:
    # discounted potential with no distance cap paid the agent to hover.
    "old_shaping":   {"DQN_SHAPING_GAMMA": "0.95", "DQN_SHAPING_SCALE": "0.1",
                      "DQN_SHAPING_CAP": "999"},
}

COMMON = {
    "DQN_ACTIONS": "UP,RIGHT,DOWN,LEFT",
    "DQN_VIEW": "ego:13",
    "DQN_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def agent_dir_for(name):
    return REPO / "agent_code" / f"dqn_{name}"


def prepare(name):
    """Give each run its own agent package so outputs never collide."""
    dest = agent_dir_for(name)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for f in ("callbacks.py", "train.py", "model.py"):
        shutil.copy2(BASE_AGENT / f, dest / f)
    return dest


def one_run(args):
    name, overrides, rounds, checkpoint_every, scenario, eps_decay = args
    agent = f"dqn_{name}"
    env = {
        **os.environ, **COMMON, **overrides,
        "DQN_EPS_DECAY": str(eps_decay),
        "DQN_CHECKPOINT_EVERY": str(checkpoint_every),
        "DQN_RESUME": "0",
    }
    cmd = [
        sys.executable, "main.py", "play", "--no-gui",
        "--agents", agent, "--train", "1",
        "--scenario", scenario, "--n-rounds", str(rounds),
    ]
    started = time.time()
    result = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True)
    minutes = (time.time() - started) / 60
    if result.returncode != 0:
        return name, minutes, result.stderr.decode()[-800:]
    return name, minutes, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=3000)
    p.add_argument("--checkpoint-every", type=int, default=250)
    p.add_argument("--scenario", default="coin-heaven")
    p.add_argument("--jobs", type=int, default=len(RUNS))
    p.add_argument("--only", nargs="*", help="run only these config names")
    args = p.parse_args()

    names = args.only or list(RUNS)
    eps_decay = max(1, args.rounds // 2)

    for name in names:
        prepare(name)

    print(f"{len(names)} configs, {args.jobs} at a time, {args.rounds} rounds each")
    print(f"epsilon decays over {eps_decay} rounds (half the run)\n")
    print("Each uses ~100 MB of replay buffer and one core.\n")

    payload = [
        (n, RUNS[n], args.rounds, args.checkpoint_every, args.scenario, eps_decay)
        for n in names
    ]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for name, minutes, error in pool.map(one_run, payload):
            status = "FAILED" if error else "done"
            print(f"{name:<14} {status:<7} {minutes:5.1f} min", flush=True)
            if error:
                print(error)

    print("\nNow measure them all frozen, in parallel, into one table:")
    print(f"  python tools/compare.py --eval-rounds 40 --jobs {args.jobs}")


if __name__ == "__main__":
    main()
