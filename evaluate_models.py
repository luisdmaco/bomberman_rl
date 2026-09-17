"""
Usage:
    python evaluate_models.py                                   # solo, classic, 100 rounds
    python evaluate_models.py --opponents rule_based_agent rule_based_agent rule_based_agent
    python evaluate_models.py --opponents peaceful_agent coin_collector_agent --sort margin
"""
import argparse, glob, json, os, shutil, subprocess, sys, tempfile

AGENT_DIR = "agent_code/q_agent"
LIVE = os.path.join(AGENT_DIR, "model.npy")


def run_eval(scenario, n_rounds, seed, opponents):
    fd, stats_path = tempfile.mkstemp(suffix=".json"); os.close(fd)
    agents_cli = ["q_agent"] + opponents
    try:
        proc = subprocess.run(
            [sys.executable, "main.py", "play", "--no-gui", "--agents", *agents_cli,
             "--scenario", scenario, "--n-rounds", str(n_rounds), "--seed", str(seed),
             "--save-stats", stats_path],
            capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()
            return None, tail[-1] if tail else f"exit {proc.returncode}"
        by_agent = json.load(open(stats_path))["by_agent"]
        q = by_agent.get("q_agent")
        if q is None:
            return None, "q_agent missing from results"
        r = q.get("rounds", n_rounds) or 1
        # duplicate opponent names get suffixed _0/_1/... by the environment,
        # so match everything that isn't q_agent rather than guessing names
        opp_scores = [v.get("score", 0) / r for k, v in by_agent.items() if k != "q_agent"]
        best_opp = max(opp_scores) if opp_scores else 0.0
        q_score = q.get("score", 0) / r

        return {
            "score": q_score, "coins": q.get("coins", 0) / r, "crates": q.get("crates", 0) / r,
            "kills": q.get("kills", 0) / r, "suicides": q.get("suicides", 0) / r,
            "steps": q.get("steps", 0) / r, "bombs": q.get("bombs", 0) / r,
            "opp_best": best_opp, "margin": q_score - best_opp,
                }, None
    except Exception as e:
        return None, repr(e)
    finally:
        os.path.exists(stats_path) and os.unlink(stats_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="classic")
    ap.add_argument("--n-rounds", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--opponents", nargs="*", default=[],
                    help="e.g rule_based_agent rule_based_agent rule_based_agent")
    ap.add_argument("--sort", default="score",
                    choices=["score", "margin","coins", "crates", "kills", "suicides"])
    ap.add_argument("--glob", default=os.path.join(AGENT_DIR, "model*.npy"))
    args = ap.parse_args()

    if len(args.opponents) > 3:
        print("at most 3 opponents (4 agents total)"); return
    
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
            results[name] = run_eval(args.scenario, args.n_rounds, args.seed, args.opponents)
    finally:
        if have_live:
            shutil.move(backup, LIVE)

    def key(item):
        st, _ = item[1]
        if st is None:
            return -1e9
        if args.sort == "suicides": return -st["suicides"]   # fewer better
        return st[args.sort]

    ranked = sorted(results.items(), key=key, reverse=True)
    tag = f"vs {'+'.join(args.opponents)}" if args.opponents else "solo"
    print(f"\n{args.scenario} {tag} | {args.n_rounds} rounds | seed {args.seed} | sorted by {args.sort}\n")
    hdr = (f"{'#':>3}  {'model':<20}{'score':>7}{'kills':>7}{'suicid':>7}"
           f"{'coins':>7}{'crates':>7}{'margin':>8}")
    print(hdr); print("-" * len(hdr))
    for i, (name, (st, err)) in enumerate(ranked, 1):
        if st is None:
            print(f"{i:>3}  {name:<22}  FAILED: {err}"); continue
        print(f"{i:>3}  {name:<20}{st['score']:>7.2f}{st['kills']:>7.3f}{st['suicides']:>7.3f}"
              f"{st['coins']:>7.2f}{st['crates']:>7.2f}{st['margin']:>+8.2f}")


if __name__ == "__main__":
    main()