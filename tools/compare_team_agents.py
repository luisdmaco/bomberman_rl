#!/usr/bin/env python3
"""Frozen, meeting-ready comparison of the team's Q agent and DQN.

Both candidates play separately against three rule_based_agent instances with
the same scenario and seed. Raw per-round rows and a machine-readable summary
are preserved under experiments/runs/. Nothing trains and no live model file is
copied or overwritten.

The default DQN is the confirmed score-profile r1250 checkpoint. The Q agent
loads agent_code/q_agent/model.npy, matching its normal submission path.
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
DEFAULT_DQN = REPO / "models/task4-score-r1250-ego13.pt"
Q_MODEL = REPO / "agent_code/q_agent/model.npy"
OPPONENTS = ["rule_based_agent", "rule_based_agent", "rule_based_agent"]


def mean_ci(values):
    values = list(values)
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def wilson_upper(successes, n, z=1.96):
    if n == 0:
        return 1.0
    p = successes / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre + half) / d


def run_frozen(agent, rounds, scenario, seed, output, dqn_model=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("DQN_")}
    if dqn_model is not None:
        env["DQN_MODEL_FILE"] = str(dqn_model)

    cmd = [
        sys.executable, "tools/evaluate3.py",
        "--agents", agent, *OPPONENTS,
        "--scenario", scenario,
        "--n-rounds", str(rounds),
        "--seed", str(seed),
        "--json", str(output),
    ]
    print(f"Evaluating {agent}: {rounds} frozen rounds...", flush=True)
    result = subprocess.run(cmd, cwd=REPO, env=env)
    if result.returncode != 0:
        raise SystemExit(f"{agent} evaluation failed with exit code {result.returncode}")

    rows = json.loads(output.read_text())
    if len(rows) != rounds:
        raise SystemExit(f"{agent}: expected {rounds} rows, found {len(rows)}")
    return rows


def summarize(rows, agent):
    opponents = [k for k in rows[0] if k not in (agent, "_steps")]
    if len(opponents) != 3:
        raise ValueError(f"{agent}: expected three opponents, found {opponents}")

    result = {"rounds": len(rows), "opponents": opponents}
    for metric in ("score", "coins", "kills", "suicides", "bombs", "invalid"):
        mean, ci = mean_ci(r[agent][metric] for r in rows)
        result[metric] = {"mean": mean, "ci95": ci}

    per_round_margin = [
        r[agent]["score"] - statistics.fmean(r[o]["score"] for o in opponents)
        for r in rows
    ]
    margin, margin_ci = mean_ci(per_round_margin)
    result["mean_opponent_margin"] = {"mean": margin, "ci95": margin_ci}

    result["paired_margins"] = {}
    for opponent in opponents:
        mean, ci = mean_ci(r[agent]["score"] - r[opponent]["score"] for r in rows)
        result["paired_margins"][opponent] = {
            "mean": mean,
            "ci95": ci,
            "lower95": mean - ci,
            "upper95": mean + ci,
        }

    strict = sum(
        r[agent]["score"] > max(r[o]["score"] for o in opponents) for r in rows
    )
    tied = sum(
        r[agent]["score"] == max(r[o]["score"] for o in opponents) for r in rows
    )
    n = len(rows)
    result["round_outcome"] = {
        "strict_first": strict,
        "tied_for_first": tied,
        "behind_top": n - strict - tied,
        "strict_first_rate": strict / n,
        "tied_for_first_rate": tied / n,
        "behind_top_rate": (n - strict - tied) / n,
    }

    suicides = sum(r[agent]["suicides"] for r in rows)
    result["suicide_upper95"] = wilson_upper(suicides, n)
    result["kill_round_rate"] = sum(r[agent]["kills"] > 0 for r in rows) / n
    result["gate4_pass"] = all(
        margin["lower95"] > 0 for margin in result["paired_margins"].values()
    )
    return result


def print_summary(results):
    print("\nFrozen team comparison")
    print("-" * 103)
    print(
        f"{'agent':<12}{'score':>16}{'vs mean RB':>18}{'strict first':>15}"
        f"{'top tie':>11}{'suicide':>12}{'kills/round':>15}{'gate 4':>10}"
    )
    print("-" * 103)
    for agent, result in results.items():
        score = result["score"]
        margin = result["mean_opponent_margin"]
        outcome = result["round_outcome"]
        print(
            f"{agent:<12}{score['mean']:>7.2f} +/- {score['ci95']:<5.2f}"
            f"{margin['mean']:>9.2f} +/- {margin['ci95']:<5.2f}"
            f"{outcome['strict_first_rate']:>14.1%}"
            f"{outcome['tied_for_first_rate']:>11.1%}"
            f"{result['suicides']['mean']:>12.1%}"
            f"{result['kill_round_rate']:>15.1%}"
            f"{('PASS' if result['gate4_pass'] else 'FAIL'):>10}"
        )

    print("\nGate 4 requires every individual DQN/Q-vs-RB 95% margin interval to be above zero.")
    print("Strict-first rate answers the separate visual question: how often is ours the sole winner?")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--scenario", default="classic")
    parser.add_argument("--dqn-model", type=Path, default=DEFAULT_DQN)
    parser.add_argument("--output-dir", type=Path, default=REPO / "experiments/runs")
    args = parser.parse_args()

    dqn_model = args.dqn_model.resolve()
    if not dqn_model.is_file():
        raise SystemExit(f"DQN checkpoint not found: {dqn_model}")
    if not Q_MODEL.is_file():
        raise SystemExit(f"Q model not found: {Q_MODEL}")
    if args.rounds < 2:
        raise SystemExit("--rounds must be at least 2 so confidence intervals are meaningful")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    raw = {
        "dqn_agent": args.output_dir / f"{stamp}-dqn-r1250-n{args.rounds}-seed{args.seed}.json",
        "q_agent": args.output_dir / f"{stamp}-q-agent-n{args.rounds}-seed{args.seed}.json",
    }

    rows = {
        "dqn_agent": run_frozen(
            "dqn_agent", args.rounds, args.scenario, args.seed, raw["dqn_agent"], dqn_model
        ),
        "q_agent": run_frozen(
            "q_agent", args.rounds, args.scenario, args.seed, raw["q_agent"]
        ),
    }
    results = {agent: summarize(agent_rows, agent) for agent, agent_rows in rows.items()}
    print_summary(results)

    summary = args.output_dir / f"{stamp}-team-comparison-n{args.rounds}-seed{args.seed}.json"
    summary.write_text(json.dumps({
        "protocol": {
            "training": False,
            "scenario": args.scenario,
            "rounds": args.rounds,
            "seed": args.seed,
            "opponents": OPPONENTS,
            "dqn_model": str(dqn_model),
            "q_model": str(Q_MODEL),
        },
        "raw_files": {name: str(path) for name, path in raw.items()},
        "results": results,
    }, indent=2))
    try:
        display_dir = args.output_dir.relative_to(REPO)
    except ValueError:
        display_dir = args.output_dir
    print(f"\nRaw rows and summary written under {display_dir}/")
    print(f"Summary: {summary.name}")


if __name__ == "__main__":
    main()
