"""
Usage:
    python evaluate_models.py                                  # loot-crate, 100 rounds, seed 42
    python evaluate_models.py --scenario classic --n-rounds 200 --sort task2
"""
import argparse, glob, json, os, shutil, subprocess, sys, tempfile

AGENT_DIR = "agent_code/q_agent"
LIVE = os.path.join(AGENT_DIR, "model.npy")


def run_eval(scenario, n_rounds, seed):
    fd, stats_path = tempfile.mkstemp(suffix=".json"); os.close(fd)
    try:
        proc = subprocess.run(
            [sys.executable, "main.py", "play", "--no-gui", "--agents", "q_agent",
             "--scenario", scenario, "--n-rounds", str(n_rounds), "--seed", str(seed),
             "--save-stats", stats_path],
            capture_output=True, text=True, timeout=2400)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()
            return None, tail[-1] if tail else f"exit {proc.returncode}"
        agg = json.load(open(stats_path))["by_agent"]["q_agent"]
        r = agg.get("rounds", n_rounds) or 1
        return {"coins": agg.get("coins", 0) / r, "crates": agg.get("crates", 0) / r,
                "suicides": agg.get("suicides", 0) / r, "steps": agg.get("steps", 0) / r,
                "bombs": agg.get("bombs", 0) / r}, None
    except Exception as e:
        return None, repr(e)
    finally:
        os.path.exists(stats_path) and os.unlink(stats_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="loot-crate")
    ap.add_argument("--n-rounds", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sort", default="coins",
                    choices=["coins", "crates", "suicides", "steps", "task2"])
    ap.add_argument("--glob", default=os.path.join(AGENT_DIR, "model*.npy"))
    args = ap.parse_args()

    candidates = sorted(glob.glob(args.glob))
    if not candidates:
        print("no models matched", args.glob); return

    backup = LIVE + ".bak"
    have_live = os.path.exists(LIVE)
    if have_live:
        shutil.copy2(LIVE, backup)
    results = {}
    try:
        for path in candidates:
            name = os.path.basename(path)
            src = backup if os.path.abspath(path) == os.path.abspath(LIVE) else path
            shutil.copy2(src, LIVE)
            print(f"  evaluating {name} ...", flush=True)
            results[name] = run_eval(args.scenario, args.n_rounds, args.seed)
    finally:
        if have_live:
            shutil.move(backup, LIVE)

    def key(item):
        st, _ = item[1]
        if st is None:
            return -1e9
        if args.sort == "suicides": return -st["suicides"]   # fewer better
        if args.sort == "steps":    return -st["steps"]       # fewer better (only vs coins)
        if args.sort == "task2":    return st["coins"] - 3 * st["suicides"]
        return st[args.sort]

    ranked = sorted(results.items(), key=key, reverse=True)
    print(f"\n{args.scenario} | {args.n_rounds} rounds | seed {args.seed} | sorted by {args.sort}\n")
    hdr = f"{'#':>3}  {'model':<22}{'coins':>8}{'crates':>8}{'suicid':>8}{'steps':>8}{'bombs':>7}"
    print(hdr); print("-" * len(hdr))
    for i, (name, (st, err)) in enumerate(ranked, 1):
        if st is None:
            print(f"{i:>3}  {name:<22}  FAILED: {err}"); continue
        print(f"{i:>3}  {name:<22}{st['coins']:>8.2f}{st['crates']:>8.2f}"
              f"{st['suicides']:>8.3f}{st['steps']:>8.1f}{st['bombs']:>7.2f}")


if __name__ == "__main__":
    main()