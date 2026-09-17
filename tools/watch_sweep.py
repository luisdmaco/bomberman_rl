#!/usr/bin/env python3
"""Show progress of a running sweep.

tools/sweep.py captures each run's output, so the tqdm bars are invisible. But
every run appends a row per round to its own training_log.csv, so progress can
be read off disk at any time, from another terminal, without disturbing
anything.

    python tools/watch_sweep.py              # one snapshot
    python tools/watch_sweep.py --watch      # refresh every 30s until done

The coins column is TRAINING-time performance, measured while the weights are
still moving. We know that overstates the frozen agent by up to 7x, so treat it
as a progress indicator and a rough between-config comparison, never as a
result. The real numbers come from tools/train_curve.py --eval-only afterwards.
"""

import argparse
import csv
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def read_runs(total_rounds):
    out = []
    for d in sorted((REPO / "agent_code").glob("dqn_*")):
        name = d.name.replace("dqn_", "")
        if name == "agent":
            continue
        log = d / "training_log.csv"
        if not log.exists():
            out.append({"name": name, "rounds": 0})
            continue
        with open(log) as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            out.append({"name": name, "rounds": 0})
            continue
        recent = rows[-100:]
        out.append({
            "name": name,
            "rounds": len(rows),
            "epsilon": float(rows[-1]["epsilon"]),
            "coins": sum(float(r["coins"]) for r in recent) / len(recent),
            "invalid": sum(float(r["invalid_actions"]) for r in recent) / len(recent),
            "stale": time.time() - log.stat().st_mtime,
        })
    return out


def snapshot(total_rounds, rate_sample=20):
    """Read twice, a few seconds apart, so we can estimate a finish time."""
    first = {r["name"]: r["rounds"] for r in read_runs(total_rounds)}
    time.sleep(rate_sample)
    runs = read_runs(total_rounds)

    print(f"{'config':<13}{'rounds':>8}{'':>6}{'eps':>7}{'coins*':>9}"
          f"{'invalid':>9}{'rounds/min':>12}{'ETA':>9}")
    print("-" * 73)
    for r in runs:
        if not r["rounds"]:
            print(f"{r['name']:<13}{'starting...':>14}")
            continue
        done = r["rounds"]
        rate = (done - first.get(r["name"], done)) / (rate_sample / 60)
        left = max(0, total_rounds - done)
        eta = f"{left / rate:.0f}m" if rate > 0.1 else ("done" if left == 0 else "  ?")
        flag = "  STALLED" if r["stale"] > 120 else ""
        print(f"{r['name']:<13}{done:>8}{100 * done / total_rounds:>5.0f}%"
              f"{r['epsilon']:>7.3f}{r['coins']:>9.1f}{r['invalid']:>9.1f}"
              f"{rate:>12.0f}{eta:>9}{flag}")

    print("\n* training-time coins, weights still moving. Overstates the frozen")
    print("  agent, which is the only number that counts. Random walk is ~18.")
    return all(r.get("rounds", 0) >= total_rounds for r in runs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=3000, help="rounds each run was given")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval", type=int, default=30)
    args = p.parse_args()

    while True:
        print(f"\n{time.strftime('%H:%M:%S')}")
        finished = snapshot(args.rounds)
        if finished:
            print("\nAll runs finished. Build the honest curves:")
            print("  python tools/train_curve.py --eval-only")
            break
        if not args.watch:
            break
        time.sleep(max(0, args.interval - 20))


if __name__ == "__main__":
    main()
