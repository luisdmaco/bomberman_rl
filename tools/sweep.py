#!/usr/bin/env python3
"""Run several training configurations at the same time, one core each.

The project spec forbids multiprocessing in the *submitted agent*, but says
explicitly that "multiprocessing or anything else that comes to your mind to
improve training is perfectly fine". This is that.

A single training run cannot use more than about one core: the game loop is
serial Python and the network is too small for extra threads to help (a second
thread buys under 10%, a second process buys a whole extra run). So the way to
use an 8-core machine is to run the whole ablation at once instead of one
experiment at a time.

Each run gets its own agent directory copy, so checkpoints, logs and curves
never collide.

    python tools/sweep.py --preset task4 --rounds 5000 --jobs 6

Two presets, because the two tasks need genuinely different settings and the
task-1 ablation has to stay reproducible for the report:

  task1  coin-heaven, solo, action space restricted to the four moves. This is
         the configuration that produced experiments/sweep_results.csv. Left
         exactly as it was; do not edit it to make a new experiment fit.
  task4  classic against three rule_based_agents, full action space, score
         reward profile. Trains from scratch, because gamma cannot be tested
         honestly by fine-tuning an existing policy: changing it rescales every
         value the old weights already encode.

This file touches nothing under agent_code/dqn_agent/. It copies the three
source files into a per-run directory and runs them there, so a sweep can run
while somebody else trains from the same commit, and so every run provably
trains the committed code rather than a local edit.

Ambient DQN_* variables are stripped before each run. A leftover DQN_GAMMA in
the shell would otherwise apply to every configuration at once and silently
turn the sweep into six copies of the same experiment.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE_AGENT = REPO / "agent_code" / "dqn_agent"
SOURCE_FILES = ("callbacks.py", "train.py", "model.py")

# Marks a directory as sweep output rather than a sweep configuration. Named so
# it still matches the agent_code/dqn_*/ gitignore rule.
ARCHIVE_PREFIX = "archive_"


PRESETS = {
    # Task 1. Reproduces experiments/sweep_results.csv. Frozen: the report
    # cites these numbers, so this preset is history, not a starting point.
    "task1": {
        "scenario": "coin-heaven",
        "opponents": [],
        "results": "sweep_results.csv",
        "common": {
            "DQN_ACTIONS": "UP,RIGHT,DOWN,LEFT",
            "DQN_VIEW": "ego:13",
            "DQN_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        "runs": {
            "baseline":    {},
            "no_shaping":  {"DQN_SHAPING": "0"},
            "global_view": {"DQN_VIEW": "global"},
            "lr_2e4":      {"DQN_LR": "0.0002"},
            "batch32":     {"DQN_BATCH": "32", "DQN_TRAIN_EVERY": "4"},
            # Reproduces the reward bug found on 2026-09-05: discounted
            # potential with no distance cap paid the agent to hover.
            "old_shaping": {"DQN_SHAPING_GAMMA": "0.95", "DQN_SHAPING_SCALE": "0.1",
                            "DQN_SHAPING_CAP": "999"},
        },
    },

    # Task 4. One factor per run, plus the control twice on different seeds.
    #
    # The second control is not a wasted slot, it is the measuring stick. The
    # r1250 selection on 2026-09-14 took the best of six checkpoints whose
    # 40-round intervals were +/- 1.0 wide, read +1.42, and came back +0.55 at
    # N=1000. Without knowing how far apart two identical configurations land,
    # no ranking of the other four means anything.
    "task4": {
        "scenario": "classic",
        "opponents": ["rule_based_agent"] * 3,
        "results": "sweep_task4_results.csv",
        "common": {
            # No DQN_ACTIONS: task 4 needs bombs. The default is all six.
            "DQN_VIEW": "ego:13",
            # Matches the current submission candidate, so the sweep is
            # measured against the thing it would replace.
            "DQN_REWARD_PROFILE": "score",
            "DQN_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        "runs": {
            "ctrl_s1":   {"DQN_SEED": "1"},
            "ctrl_s2":   {"DQN_SEED": "2"},
            # Effective horizon 20 -> 100 steps. The bomb-to-coin causal chain
            # is 10 to 30 steps long; at 0.95 its far end arrives discounted to
            # 0.36. agent_code/dqn_agent/README.md has said "raise for tasks 3
            # and 4" since the agent was written, and it never was raised.
            "gamma99":   {"DQN_SEED": "1", "DQN_GAMMA": "0.99"},
            # Implemented and verified by tools/test_augment.py, never once
            # switched on in a task-4 run.
            "augment":   {"DQN_SEED": "1", "DQN_AUGMENT": "1"},
            # 1000 gradient steps x train_every 16 = 16000 environment steps per
            # target sync, roughly 60 to 100 syncs in a whole run. Bootstrapping
            # a longer horizon through a target that stale is slow.
            "target250": {"DQN_SEED": "1", "DQN_TARGET_UPDATE": "250"},
            # The one piece of hard evidence available: in coin-heaven the
            # global view overtakes ego:13 after ~1000 rounds (47.9 against
            # 28.3 at r2750) while ego degrades. A 13x13 window on a 17x17
            # board cannot see the opponents task 4 is scored against.
            "global":    {"DQN_SEED": "1", "DQN_VIEW": "global"},
        },
    },
}


def preset(name):
    try:
        return PRESETS[name]
    except KeyError:
        raise SystemExit(f"unknown preset {name!r}; choose {', '.join(PRESETS)}")


def clean_env():
    """os.environ without any ambient DQN_* override."""
    return {k: v for k, v in os.environ.items() if not k.startswith("DQN_")}


def env_for(preset_name, run_name, extra=None):
    """The exact environment one configuration is trained and evaluated with.

    compare.py imports this. Evaluating dqn_global without DQN_VIEW=global
    would build a network of the wrong shape and fail to load its own weights.
    """
    cfg = preset(preset_name)
    env = dict(cfg["common"])
    env.update(cfg["runs"].get(run_name, {}))
    env.update(extra or {})
    return env


def agent_dir_for(name):
    return REPO / "agent_code" / f"dqn_{name}"


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                             capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except OSError:
        return "unknown"


def prepare(name, force=False):
    """Give each run its own agent package so outputs never collide.

    An existing directory is moved aside, not deleted. On 2026-09-09 a rerun
    overwrote another run's weights and its checkpoints were gone; the same
    lesson is written into train_curve.archive_checkpoints. --force restores
    the old delete-and-recreate behaviour for when the disk is actually full.
    """
    dest = agent_dir_for(name)
    if dest.exists():
        if force:
            shutil.rmtree(dest)
        else:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            archive = agent_dir_for(f"{ARCHIVE_PREFIX}{stamp}_{name}")
            dest.rename(archive)
            print(f"  {dest.name} existed; moved to {archive.name}/")
    dest.mkdir(parents=True)
    for f in SOURCE_FILES:
        shutil.copy2(BASE_AGENT / f, dest / f)
    return dest


def one_run(args):
    (name, env_extra, rounds, checkpoint_every, scenario, opponents,
     eps_decay, log_root) = args
    agent = f"dqn_{name}"
    env = {
        **clean_env(), **env_extra,
        "DQN_EPS_DECAY": str(eps_decay),
        "DQN_CHECKPOINT_EVERY": str(checkpoint_every),
        "DQN_RESUME": "0",
    }
    # Its own log directory. main.py defaults --log-dir to REPO/logs, so six
    # parallel runs all append to one game.log: the lines interleave mid-write
    # and a crashed run cannot be told apart from the five healthy ones. It is
    # also a single contended file, and in this checkout a OneDrive-synced one.
    log_dir = Path(log_root) / f"sweep-{name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "main.py", "play", "--no-gui",
        "--agents", agent, *opponents, "--train", "1",
        "--scenario", scenario, "--n-rounds", str(rounds),
        "--log-dir", str(log_dir),
    ]
    started = time.time()
    result = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True)
    minutes = (time.time() - started) / 60
    if result.returncode != 0:
        return name, minutes, result.stderr.decode()[-800:]
    return name, minutes, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="task4", choices=sorted(PRESETS),
                   help="which grid to run (default: task4)")
    p.add_argument("--rounds", type=int, default=5000)
    p.add_argument("--checkpoint-every", type=int, default=250)
    p.add_argument("--scenario", help="override the preset's scenario")
    p.add_argument("--opponents", nargs="*",
                   help="override the preset's opponents; pass with no names for solo")
    p.add_argument("--eps-decay", type=int,
                   help="rounds over which epsilon anneals (default: half the run)")
    p.add_argument("--jobs", type=int, default=6,
                   help="parallel runs; keep at least two cores free for evaluation")
    p.add_argument("--only", nargs="*", help="run only these config names")
    p.add_argument("--log-dir", default=str(REPO / "logs"),
                   help="base directory for per-run game logs (default: logs/)")
    p.add_argument("--force", action="store_true",
                   help="delete existing run directories instead of archiving them")
    args = p.parse_args()

    cfg = preset(args.preset)
    scenario = args.scenario or cfg["scenario"]
    opponents = cfg["opponents"] if args.opponents is None else args.opponents
    names = args.only or list(cfg["runs"])
    unknown = [n for n in names if n not in cfg["runs"]]
    if unknown:
        raise SystemExit(f"not in preset {args.preset!r}: {', '.join(unknown)}")

    eps_decay = args.eps_decay or max(1, args.rounds // 2)
    commit = git_commit()

    print(f"preset {args.preset}: {len(names)} configs, {args.jobs} at a time, "
          f"{args.rounds} rounds each")
    print(f"scenario {scenario}"
          + (f" against {', '.join(opponents)}" if opponents else " (solo)"))
    print(f"epsilon decays over {eps_decay} rounds; training code at commit {commit[:8]}")
    print("Each uses ~100 MB of replay buffer and one core.\n")

    for name in names:
        prepare(name, force=args.force)

    payload = []
    manifest = {"preset": args.preset, "commit": commit, "scenario": scenario,
                "opponents": opponents, "rounds": args.rounds,
                "eps_decay": eps_decay,
                "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "runs": {}}
    for n in names:
        env_extra = env_for(args.preset, n)
        manifest["runs"][n] = env_extra
        payload.append((n, env_extra, args.rounds, args.checkpoint_every,
                        scenario, opponents, eps_decay, args.log_dir))

    # Written before training starts, so an interrupted sweep still says what it
    # was running. A teammate's separate run can only be compared against ctrl
    # if both the commit and the overrides are on record.
    stamp = time.strftime("%Y-%m-%d")
    manifest_path = (REPO / "experiments" / "runs"
                     / f"{stamp}-{args.preset}-sweep-manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Manifest: {manifest_path.relative_to(REPO)}\n")

    started = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for name, minutes, error in pool.map(one_run, payload):
            status = "FAILED" if error else "done"
            print(f"{name:<14} {status:<7} {minutes:5.1f} min", flush=True)
            if error:
                print(error)

    print(f"\nWhole sweep: {(time.time() - started) / 60:.1f} min")
    print("\nNow measure them all frozen, in parallel, into one table:")
    print(f"  python tools/compare.py --preset {args.preset} "
          f"--eval-rounds 60 --jobs {args.jobs}")


if __name__ == "__main__":
    main()
