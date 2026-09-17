#!/usr/bin/env python3
"""Evaluate every sweep configuration frozen, and put them in one table.

Each config's checkpoints are replayed with training switched off, which is the
only measurement that describes the agent that would be submitted. Configs are
evaluated in parallel, one core each, the same way they were trained.

    python tools/compare.py --preset task4 --eval-rounds 60 --jobs 6

Critically, each config is evaluated with the SAME environment overrides it was
trained with, taken from sweep.env_for. Evaluating dqn_global without
DQN_VIEW=global would build a network of the wrong shape and fail to load its
own weights.

## Configs are ranked on a pooled tail, not on their best checkpoint

`max` over a curve of noisy checkpoints is a biased estimator, and this project
has already paid for that once: on 2026-09-14 the best of six 40-round
checkpoints read a score margin of +1.42, and the same weights came back at
+0.55 over N=1000. With six configurations to rank, six independent chances to
draw a lucky checkpoint make it worse, not better.

So the ranking column is the mean over the last --last-k checkpoints, which at
the default settings pools 5 x 60 = 300 frozen rounds per configuration. The
per-checkpoint best is still printed and still written to the CSV, because the
checkpoint that is eventually submitted has to be chosen from somewhere, but it
is not what decides which hyperparameter won.

A configuration only counts as a real improvement if it beats the control by
more than the two control seeds differ from each other. That number is printed
under the table; it is the whole reason ctrl is in the grid twice.
"""

import argparse
import csv
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from evaluate import run as run_stats, mean_ci  # noqa: E402
from sweep import ARCHIVE_PREFIX, clean_env, env_for, preset  # noqa: E402
from train_curve import evaluate_checkpoint_versus  # noqa: E402

SOLO_COLUMNS = ["config", "rounds_trained", "frozen_coins", "ci95",
                "mean_steps", "invalid_per_round"]
VERSUS_COLUMNS = ["config", "rounds_trained", "frozen_score", "score_ci95",
                  "score_margin", "margin_ci95", "frozen_coins", "ci95",
                  "suicide_rate", "kill_in_round", "mean_steps",
                  "invalid_per_round"]


def evaluate_one_solo(path, agent, rounds, scenario, env):
    stats = run_stats([agent], scenario, rounds, seed=None,
                      extra_env={**env, "DQN_MODEL_FILE": str(path)})
    per_round = stats["by_round"].values()
    coins = [r.get("coins", 0) for r in per_round]
    steps = [r.get("steps", 0) for r in per_round]
    invalid = stats["by_agent"].get(agent, {}).get("invalid", 0)
    mean, ci = mean_ci(coins)
    return dict(score=mean, score_ci=ci, margin=0.0, margin_ci=0.0,
                coins=mean, ci=ci, suicide=0.0, killed=0.0,
                steps=sum(steps) / max(1, len(steps)),
                invalid=invalid / max(1, len(steps)))


def evaluate_config(job):
    """One configuration, every checkpoint, in checkpoint order.

    Each checkpoint is measured in a fresh interpreter (evaluate3.py) because
    callbacks.MODEL_FILE is a module-level constant frozen at first import: a
    loop over checkpoints inside one process measures the first one every time
    and draws a flat curve. train_curve.py carries the same warning.
    """
    name, preset_name, rounds, scenario, opponents = job
    agent = f"dqn_{name}"
    agent_dir = REPO / "agent_code" / agent
    checkpoints = sorted((agent_dir / "checkpoints").glob("dqn-r*.pt"))
    if not checkpoints:
        return name, [], "no checkpoints"

    env = env_for(preset_name, name)
    curve = []
    for path in checkpoints:
        trained = int(path.stem.split("-r")[1])
        try:
            if opponents:
                m = evaluate_checkpoint_versus(path, rounds, scenario, opponents,
                                               env, agent=agent)
            else:
                m = evaluate_one_solo(path, agent, rounds, scenario, env)
        except SystemExit as exc:
            return name, curve, str(exc)
        curve.append((trained, m))

    columns = VERSUS_COLUMNS[1:] if opponents else SOLO_COLUMNS[1:]
    with open(agent_dir / "eval_curve.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        for trained, m in curve:
            w.writerow(row_for(trained, m, opponents))
    return name, curve, None


def row_for(trained, m, opponents):
    if opponents:
        return [trained, round(m["score"], 3), round(m["score_ci"], 3),
                round(m["margin"], 3), round(m["margin_ci"], 3),
                round(m["coins"], 2), round(m["ci"], 2),
                round(m["suicide"], 4), round(m["killed"], 4),
                round(m["steps"], 1), round(m["invalid"], 2)]
    return [trained, round(m["coins"], 2), round(m["ci"], 2),
            round(m["steps"], 1), round(m["invalid"], 2)]


def display_path(path):
    """Repo-relative when it is inside the repo, absolute otherwise. --out may
    legitimately point anywhere, and relative_to raises rather than falling back."""
    try:
        return path.relative_to(REPO)
    except ValueError:
        return path


def pooled_tail(curve, key, last_k):
    """Mean of `key` over the last k checkpoints.

    Equal weight per checkpoint, which is right only because every checkpoint
    is measured over the same number of rounds. If --eval-rounds ever varies
    per checkpoint this has to become a weighted mean.
    """
    tail = curve[-last_k:] if last_k else curve
    return statistics.fmean([m[key] for _, m in tail]), len(tail)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="task4",
                   help="which sweep grid these directories came from")
    p.add_argument("--eval-rounds", type=int, default=60)
    p.add_argument("--scenario", help="override the preset's scenario")
    p.add_argument("--opponents", nargs="*",
                   help="override the preset's opponents; pass with no names for solo")
    p.add_argument("--last-k", type=int, default=5,
                   help="checkpoints pooled into the ranking metric (0 = all)")
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--only", nargs="*")
    p.add_argument("--out", help="results CSV (default: the preset's own file)")
    args = p.parse_args()

    cfg = preset(args.preset)
    scenario = args.scenario or cfg["scenario"]
    opponents = cfg["opponents"] if args.opponents is None else args.opponents
    results_path = Path(args.out) if args.out else REPO / "experiments" / cfg["results"]

    names = args.only or [
        d.name[len("dqn_"):] for d in sorted((REPO / "agent_code").glob("dqn_*"))
        if d.name != "dqn_agent"
        and not d.name[len("dqn_"):].startswith(ARCHIVE_PREFIX)
        and (d / "checkpoints").is_dir()
    ]
    names = [n for n in names if n in cfg["runs"]]
    if not names:
        raise SystemExit(f"no {args.preset} configs with checkpoints found")

    # An ambient DQN_VIEW from an earlier shell command would be inherited by
    # every evaluation subprocess and silently measure the wrong architecture.
    # Read BEFORE clearing: clean_env() reads os.environ, so computing it as the
    # argument to update() would hand back the already-emptied mapping and drop
    # PATH along with everything else.
    cleaned = clean_env()
    os.environ.clear()
    os.environ.update(cleaned)

    print(f"preset {args.preset}: {len(names)} configs, {args.eval_rounds} rounds "
          f"per checkpoint, training OFF, {args.jobs} at a time")
    print(f"scenario {scenario}"
          + (f" against {', '.join(opponents)}" if opponents else " (solo)") + "\n")
    started = time.time()

    ranking_key = "margin" if opponents else "coins"
    results = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        jobs = [(n, args.preset, args.eval_rounds, scenario, opponents) for n in names]
        for name, curve, error in pool.map(evaluate_config, jobs):
            if error:
                print(f"{name:<14} FAILED: {error}")
                continue
            results[name] = curve
            pooled, k = pooled_tail(curve, ranking_key, args.last_k)
            print(f"{name:<14} done, pooled {ranking_key} over last {k} "
                  f"checkpoints: {pooled:+.3f}", flush=True)

    if not results:
        raise SystemExit("every configuration failed")

    print(f"\nEvaluated in {(time.time() - started) / 60:.1f} min\n")

    results_path.parent.mkdir(exist_ok=True)
    columns = VERSUS_COLUMNS if opponents else SOLO_COLUMNS
    # Rows for configurations this invocation did NOT evaluate are carried over
    # rather than dropped. Opening in "w" and writing only `results` meant that
    # a --only rerun of one arm silently deleted every other arm's measurements
    # from the file, which is hours of evaluation gone with no error.
    kept = []
    if results_path.exists():
        with open(results_path, newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header == columns:
                kept = [r for r in reader if r and r[0] not in results]
            elif header:
                stamp = time.strftime("%Y%m%d-%H%M%S")
                superseded = results_path.with_name(f"{results_path.stem}.{stamp}.csv")
                results_path.rename(superseded)
                print(f"Existing {results_path.name} had different columns; "
                      f"kept as {superseded.name}")
    with open(results_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        for row in kept:
            w.writerow(row)
        for name, curve in results.items():
            for trained, m in curve:
                w.writerow([name, *row_for(trained, m, opponents)])
    if kept:
        carried = sorted({r[0] for r in kept})
        print(f"Carried over {len(kept)} rows from {', '.join(carried)}")
    print(f"Wrote {display_path(results_path)}\n")

    ranked = sorted(results.items(),
                    key=lambda kv: -pooled_tail(kv[1], ranking_key, args.last_k)[0])

    if opponents:
        print(f"{'config':<12}{'pooled':>9}{'best ck':>9}{'@rounds':>9}"
              f"{'score':>8}{'suicide':>9}{'kill/rd':>9}")
        print("-" * 65)
        for name, curve in ranked:
            pooled, _ = pooled_tail(curve, "margin", args.last_k)
            best_round, best = max(curve, key=lambda r: r[1]["margin"])
            tail_score, _ = pooled_tail(curve, "score", args.last_k)
            tail_suicide, _ = pooled_tail(curve, "suicide", args.last_k)
            tail_kill, _ = pooled_tail(curve, "killed", args.last_k)
            print(f"{name:<12}{pooled:>+9.3f}{best['margin']:>+9.3f}{best_round:>9}"
                  f"{tail_score:>8.2f}{tail_suicide:>8.1%}{tail_kill:>9.1%}")
    else:
        print(f"{'config':<12}{'pooled':>9}{'best ck':>9}{'@rounds':>9}"
              f"{'steps':>8}{'invalid':>9}")
        print("-" * 56)
        for name, curve in ranked:
            pooled, _ = pooled_tail(curve, "coins", args.last_k)
            best_round, best = max(curve, key=lambda r: r[1]["coins"])
            final = curve[-1][1]
            print(f"{name:<12}{pooled:>9.2f}{best['coins']:>9.2f}{best_round:>9}"
                  f"{final['steps']:>8.0f}{final['invalid']:>9.2f}")

    # The measuring stick. Two configurations that differ only in their seed
    # bound how much of any gap above is real.
    ctrls = [n for n in results if n.startswith("ctrl")]
    if len(ctrls) >= 2:
        pooled = {n: pooled_tail(results[n], ranking_key, args.last_k)[0] for n in ctrls}
        spread = max(pooled.values()) - min(pooled.values())
        print(f"\nControl seed spread ({', '.join(ctrls)}): {spread:.3f} {ranking_key}.")
        print("Treat any configuration within that distance of ctrl as unseparated,")
        print("however the table happens to have ordered it.")
    else:
        print("\nNo second control seed in these results, so the table has no scale:")
        print("nothing here says how much of a gap is the hyperparameter and how")
        print("much is the seed. Run ctrl_s1 and ctrl_s2 before believing a ranking.")

    print("\nShortlist only. Confirm the winner at N=1000 on an unseen world seed")
    print("before it goes in the report or the submission:")
    print("  python tools/evaluate3.py --agents dqn_agent rule_based_agent "
          "rule_based_agent rule_based_agent \\")
    print("      --scenario classic --n-rounds 1000 --seed <unused seed>")


if __name__ == "__main__":
    main()
