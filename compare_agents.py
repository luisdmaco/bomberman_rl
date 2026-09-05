import json
import sys
import statistics as st

# usage: compare the stats from 2 agents

def load_rounds(path):
    with open(path) as f:
        data = json.load(f)
    return list(data["by_round"].values())


def summarize(name, rounds):
    coins = [r["coins"] for r in rounds]
    steps = [r["steps"] for r in rounds]
    efficiency = [c / s for c, s in zip(coins, steps) if s > 0]

    print(f"--- {name} ({len(rounds)} rounds) ---")
    print(f"  coins/round : mean={st.mean(coins):.2f}  stdev={st.pstdev(coins):.2f}  min={min(coins)}  max={max(coins)}")
    print(f"  steps/round : mean={st.mean(steps):.2f}  stdev={st.pstdev(steps):.2f}  min={min(steps)}  max={max(steps)}")
    print(f"  coins/step  : mean={st.mean(efficiency):.4f}")
    full_clear = sum(1 for c in coins if c >= 50)
    print(f"  rounds with all 50 coins collected: {full_clear}/{len(rounds)}")
    print()
    return {
        "mean_coins": st.mean(coins),
        "mean_steps": st.mean(steps),
        "mean_efficiency": st.mean(efficiency),
    }


if __name__ == "__main__":
    q_path, baseline_path = sys.argv[1], sys.argv[2]

    q_stats = summarize("q_agent", load_rounds(q_path))
    base_stats = summarize("coin_collector_agent (baseline)", load_rounds(baseline_path))

    print("--- gap (q_agent - baseline) ---")
    print(f"  coins/round : {q_stats['mean_coins'] - base_stats['mean_coins']:+.2f}")
    print(f"  steps/round : {q_stats['mean_steps'] - base_stats['mean_steps']:+.2f}  (negative = q_agent faster)")
    print(f"  coins/step  : {q_stats['mean_efficiency'] - base_stats['mean_efficiency']:+.4f}  (negative = baseline more efficient)")