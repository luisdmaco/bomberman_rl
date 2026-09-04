#!/usr/bin/env python3
"""Re-measure the best checkpoints over many more rounds and check gate 1.

The sweep evaluates each checkpoint over 40 rounds, which gives a confidence
interval of roughly +/- 2 coins. That is fine for ranking but far too wide to
claim a checkpoint clears a threshold of 45. Picking the maximum of 72 noisy
measurements also biases upward: the winner is partly winning on luck.

So the candidates get re-measured on a fresh set of rounds, which is the honest
way to confirm a threshold.

    python tools/confirm.py --n-rounds 200 --top 5

Each candidate is evaluated with the environment overrides it was trained with,
read from sweep.py, so a config trained on the global board is not evaluated
with an egocentric network.
"""

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from evaluate import run as run_stats, mean_ci  # noqa: E402
from sweep import RUNS, COMMON  # noqa: E402

SWEEP = REPO / "experiments" / "sweep_results.csv"
OUT = REPO / "experiments" / "gate1_confirmation.csv"

GATE_COINS = 45
GATE_INVALID = 0.0


def measure(job):
    config, rounds_trained, n_rounds, scenario = job
    agent = f"dqn_{config}"
    ckpt = REPO / "agent_code" / agent / "checkpoints" / f"dqn-r{int(rounds_trained):06d}.pt"
    if not ckpt.exists():
        return config, rounds_trained, None, f"missing {ckpt.name}"

    env = {**COMMON, **RUNS.get(config, {}), "DQN_MODEL_FILE": str(ckpt)}
    stats = run_stats([agent], scenario, n_rounds, seed=None, extra_env=env)
    per_round = list(stats["by_round"].values())
    coins = [r.get("coins", 0) for r in per_round]
    steps = [r.get("steps", 0) for r in per_round]
    invalid = stats["by_agent"].get(agent, {}).get("invalid", 0) / max(1, len(per_round))
    mean, ci = mean_ci(coins)
    return config, rounds_trained, (mean, ci, sum(steps) / len(steps), invalid), None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-rounds", type=int, default=200)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--scenario", default="coin-heaven")
    p.add_argument("--jobs", type=int, default=5)
    args = p.parse_args()

    rows = list(csv.DictReader(open(SWEEP)))
    ranked = sorted(rows, key=lambda r: -float(r["frozen_coins"]))
    # One candidate per config: its own best checkpoint.
    seen, candidates = set(), []
    for r in ranked:
        if r["config"] in seen:
            continue
        seen.add(r["config"])
        candidates.append((r["config"], int(r["rounds_trained"])))
        if len(candidates) >= args.top:
            break

    print(f"Confirming {len(candidates)} checkpoints over {args.n_rounds} rounds each "
          f"(the sweep used 40).\n")
    for c, r in candidates:
        print(f"  {c} @ {r} rounds")
    print()

    jobs = [(c, r, args.n_rounds, args.scenario) for c, r in candidates]
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for config, trained, values, error in pool.map(measure, jobs):
            if error:
                print(f"{config:<14} FAILED: {error}")
                continue
            results.append((config, trained, *values))

    print(f"{'config':<14}{'@rounds':>9}{'coins':>8}{'95% CI':>16}"
          f"{'steps':>8}{'invalid':>9}  gate 1")
    print("-" * 76)
    for config, trained, mean, ci, steps, invalid in sorted(results, key=lambda r: -r[2]):
        lo, hi = mean - ci, mean + ci
        # Only a claim if the whole interval clears the bar.
        passed = lo >= GATE_COINS and invalid <= GATE_INVALID
        marginal = mean >= GATE_COINS and not passed
        verdict = "PASS" if passed else ("marginal" if marginal else "fail")
        print(f"{config:<14}{trained:>9}{mean:>8.2f}  [{lo:>5.1f}, {hi:>5.1f}]"
              f"{steps:>8.0f}{invalid:>9.2f}  {verdict}")

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "rounds_trained", "coins", "ci95", "mean_steps",
                    "invalid_per_round", "n_rounds"])
        for config, trained, mean, ci, steps, invalid in results:
            w.writerow([config, trained, round(mean, 2), round(ci, 2),
                        round(steps, 1), round(invalid, 2), args.n_rounds])

    print(f"\nPASS requires the whole confidence interval above {GATE_COINS} coins")
    print(f"and no invalid actions. 'marginal' means the mean clears it but the")
    print(f"interval does not, which is not yet a claim you can make in a report.")
    print(f"\nWritten to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
