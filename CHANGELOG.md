# Changelog — since `c0f36d7` (add rewards + training log)

Scope: task 3 (fighting `peaceful_agent`/`coin_collector_agent`) development.
Diagnosed a Q-learning maximization-bias divergence, fixed it two independent
ways (mask + target network), and added opponent-directed reward shaping.
Covers every file `git diff c0f36d7` reports as changed, plus the untracked
artifacts sitting next to them.

**Correction (verified after the commits below landed)**: the original
version of this file cautioned that `AMBUSH_READY`/`WIN_ROUND` were added
*after* the successful run, based on `train.py`'s file mtime being newer
than the other artifacts. That was wrong -- mtimes aren't reliable evidence
(touching/reformatting a file changes them without changing behaviour).
Proof: `training_log.csv`'s `win` column (`round own_score > max(opponent
scores)`) can only be written by the current `end_of_round`/`write_round_log`
code, which is the same edit that adds `WIN_ROUND`/`AMBUSH_READY`. Reading it
with `csv.DictReader` (the file uses CRLF line endings, which silently broke
a naive `awk` field check): **2474 / 3000 rounds (82.5%) are wins** against
`peaceful_agent` + `coin_collector_agent`. `old_models/task3_better/training_log.csv`
is a byte-identical copy of the same 3000 rounds, not a separate/earlier
run -- it was archived there as task-4 prep, not because it was superseded.
So: the full current code, `P5` events included, is already validated by
this run. No retraining needed before moving on.

---

## `agent_code/q_agent/q_model.py` — FIRST FIX

```diff
-    def update(self, features, action, td_target, lr):
+    def update(self, features, action, td_target, lr, max_norm=10.0):
         pred = self.weights[action] @ features
         self.weights[action] += lr * (td_target - pred) * features
+        norm = np.linalg.norm(self.weights[action])
+        if norm > max_norm:
+            self.weights[action] *= max_norm / norm
```

**Why**: diagnosed via `training_log.csv` — a 5000-round run showed `kills`
and `reward` getting *worse* over training while `WAIT`'s weight vector grew
unbounded (L2 norm 6.9 → 15.0, linearly, no sign of saturating) while every
other action stayed in the 4–7 range. This is vanilla Q-learning's
**maximization bias**: `max_a Q(s',a)` picks and evaluates with the same
weights, so an action whose next state looks like its current state (`WAIT`,
once opponents make it legal in most non-emergency states) echoes any
positive estimation error straight back into itself every update.

**Effect measured**: growth changed from unbounded/linear to decelerating
toward the 10.0 ceiling (deltas per 250 rounds: `+2.0, +2.0, ... ` before →
`+1.79, +1.40, +1.12, +0.72, +0.39` after). Stopped the blowup; did not by
itself stop `WAIT` from still growing disproportionately relative to other
actions — see the two fixes below.

---

## `agent_code/q_agent/features.py` — `get_action_mask` (tagged `P2`)

Three changes to the same function:

**1. Voluntary `WAIT` banned outright**, at the very top:
```python
mask['WAIT'] = False
```
Only re-enabled by the empty-mask fallback (change 3) as a genuine last
resort. Rationale: `WAIT` was the #1 self-inflicted-death driver (standing
next to your own ticking bomb), and — per the fix above — a linear model has
no reliable way to learn "when is waiting strategically fine" without the
same estimator that's supposed to *evaluate* that judgment also being the one
whose bias caused the problem in the first place. Removing the option is
cheaper and more robust than continuing to fight the bias through reward
tuning.

**2. Escape-pruning rewritten** to never leave the agent with zero movement
options when at least one *doomed* move still exists:
```python
if bombs:
    movable = [a for a in directions if mask[a]]
    if movable:
        doomed = [a for a in movable if not escape_exists(...)]
        survivors = [a for a in movable if a not in doomed]
        if survivors:
            for action in doomed:
                mask[action] = False
        else:
            # every move is doomed: run toward whichever blast arrives LAST
            # instead of letting argmax pick one at random
            def _deadline(a):
                d = danger_at(directions[a], dmap)
                return 99 if d is None else d
            best_t = max(_deadline(a) for a in movable)
            for action in movable:
                if _deadline(action) < best_t:
                    mask[action] = False
```
Previously (`len(movable) > 1`), a agent down to exactly one legal
direction could still have it pruned by the escape check, handing control to
the empty-mask fallback even when a real (if losing) move existed. Now:
prune doomed moves only while a survivor remains; if none survive, keep the
move that buys the most time instead of leaving the choice to chance.

**3. Empty-mask fallback rewritten** to pick the least-dangerous concrete
move rather than defaulting to `WAIT`:
```python
if not any(mask.values()):
    safe_moves = [a for a, pos in directions.items()
        if get_walkable(field, pos[0], pos[1], bombs, others)
        and danger_at(pos, dmap) != 0]
    if safe_moves:
        best = max(safe_moves, key=lambda a: 99 if danger_at(directions[a], dmap) is None
                    else danger_at(directions[a], dmap))
        mask = {a: False for a in ACTIONS}
        mask[best] = True
    else:
        mask['WAIT'] = True
```
Only reaches `mask['WAIT'] = True` when literally no direction survives even
this weaker "not exploding on this exact step" check — genuinely last-resort.

---

## `agent_code/q_agent/train.py` — SECOND/THIRD/FOURTH FIX + `P5`

**Imports / logging**: added `import copy`; imported `opponent_in_blast` from
`features.py`; extended `LOG_COLUMNS` with `opp_max_score`, `win` so
`training_log.csv` can report a per-round win/loss against the toughest
opponent present.

**New custom events**: `WAITED_WITH_OPPONENTS`, `AMBUSH_READY`, `WIN_ROUND`.

**`setup_training` — SECOND FIX (lr retune) + THIRD FIX (target network)**:
```python
self.lr = 0.005          # was 0.01
self.lr_min = 0.0005     # was 0.001
self.lr_decay = 0.999    # was 0.9995 — reaches the floor sooner

self.target_model = copy.deepcopy(self.model)
self.target_sync_every = 100
```
The target network bootstraps the TD target off a 100-round-stale copy of
the weights instead of the ones actively being updated — the standard DQN
technique for breaking exactly the self-reinforcing `max`-bootstrap loop
described under `q_model.py` above. Measured effect (training log,
before/after, first-250 vs last-250 rounds):

| | before (no target net) | after |
|---|---:|---:|
| kills | −41% | −7% |
| reward | −49% | +7% |
| coins/crates | declining | +12% |

Real stabilization, not a full fix — `trapped_opponent` (−19%) and
`bombed_opponent` (−27%) kept declining more gently, and `suicides`/`got_killed`
rose (+47%/+41%) alongside more aggressive engagement. This motivated `P2`
above (ban `WAIT` structurally) rather than chasing the target-network sync
interval further.

**`game_events_occurred`**:
- `WAITED_WITH_OPPONENTS` fires whenever `self_action == 'WAIT'` and any
  opponent is on the board (belt-and-suspenders alongside the `P2` mask ban —
  covers the rare last-resort case the mask still allows through).
- `AMBUSH_READY` (tagged `P5`) fires when, in the *new* state, a bomb is
  available, an opponent is standing in what would be this tile's blast, and
  `escape_exists` confirms dropping now wouldn't be suicide — "directed
  bombing readiness," distinct from `BOMBED_CRATES`/`BOMBED_OPPONENT` which
  only fire once a bomb has actually been dropped.
- Bootstrap target line changed:
  ```python
  future_q = self.target_model.predict(new_f)   # was self.model.predict(new_f)
  ```

**`end_of_round`**:
- `WIN_ROUND` (tagged `P5`) appended when `last_game_state['self'][1]` (own
  score) exceeds every opponent's score at round end — an auxiliary,
  training-only signal for the actual task-3 goal (winning), not just farming
  crates/coins.
- Target network synced every `target_sync_every` (100) rounds:
  ```python
  if last_game_state['round'] % self.target_sync_every == 0:
      self.target_model = copy.deepcopy(self.model)
  ```

**`write_round_log`**: now also writes `opp_max_score` and `win` (1/0) per round.

**`reward_from_events`** — value changes:

| Event | Before | After |
|---|---:|---:|
| `KILLED_OPPONENT` | 5 | 6 |
| `IN_DANGER` | −0.05 | −0.15 |
| `ESCAPED_DANGER` | 0.4 | 0.5 |
| `BOMBED_OPPONENT` | 0.5 | 0.8 |
| `TRAPPED_OPPONENT` | **0.0** | **2.0** |
| `MOVED_CLOSER_TO_OPPONENT` | 0.05 | 0.15 |
| `MOVED_FARTHER_FROM_OPPONENT` | −0.05 | −0.15 |
| `WAITED_WITH_OPPONENTS` | (new) | −0.1 |
| `AMBUSH_READY` | (new) | 0.3 |
| `WIN_ROUND` | (new) | 1.5 |
| `STEP_PENALTY` | −0.02 | −0.03 |

`TRAPPED_OPPONENT` is the single most important number in this table: it's
the one event that actually predicts a kill (a bomb the opponent cannot
escape), and it had been sitting at exactly `0.0` for two prior full training
runs (5000 and 3000 rounds) — meaning it fired hundreds of times with zero
training signal. This was isolated from `BOMBED_OPPONENT` deliberately (an
earlier attempt raised both together and lost twice, confounding which
change was responsible).

---

## `evaluate_models.py` — multi-agent evaluation support

- New `--opponents` flag (0–3 names), builds `--agents q_agent <opponents...>`
  for the subprocess instead of hardcoding solo play.
- `run_eval` now reads every key in `by_agent` except `q_agent` as an
  opponent (handles the environment's `_0`/`_1`/`_2` suffixing of duplicate
  opponent names) and reports `score`, `kills`, and `margin` (own score minus
  the strongest opponent's, per round) alongside the existing `coins`,
  `crates`, `suicides`.
- Default `--scenario` changed `loot-crate` → `classic`; default `--sort`
  changed `coins` → `score` (the actual game score — only `COIN_COLLECTED`
  and `KILLED_OPPONENT` count toward it, so `coins`/`crates` alone
  overweight economy over combat once opponents are on the board).
- Dropped the `task2`/`steps` sort options (superseded by `score`/`margin`).

---

## Data / artifacts (not code)

- **`agent_code/q_agent/model.npy` deleted.** Forces a fresh training run
  under the corrected code rather than resuming or evaluating a stale
  pre-fix save.
- **`agent_code/q_agent/model_best.npy` added** (untracked) — the current
  best checkpoint kept as the working reference outside the `old_models/`
  archive. Its `WAIT` weight (L2 6.38) still sits ~2.3x above the other
  actions' — expected given the fixes above only *stop the bias from
  mattering* (the mask keeps `WAIT` from being selected in normal play), they
  don't retroactively shrink weight already accumulated by an earlier point
  in training.
- **`agent_code/q_agent/training_log.csv` added** (untracked, 3000 rows) —
  per-round diagnostics (`kills`, `trapped_opponent`, `bombed_opponent`,
  `suicides`, `got_killed`, `reward`, ...) for the current/latest run.
- **`agent_code/q_agent/resul_of_eval_classic.txt` added** (untracked) — a
  running log of `evaluate_models.py` output across every attempt this task,
  ending in:
  ```
  LAST TRY AND SUCCESSFUL:
  classic vs peaceful_agent+coin_collector_agent | 100 rounds | seed 42
  #1  model_round1000.npy   score 9.78  kills 0.830  suicides 0.290  margin +5.14
  ```
  the best measured result so far — confirmed (see the correction note at
  the top of this file) to already reflect the full current code, `P5`
  events included: 82.5% of this run's 3000 rounds were outright wins.
- **`agent_code/q_agent/old_models/`** (gitignored, not part of any commit
  here) — `task2/` holds every task-2 checkpoint. `task3_better/` is a
  backup of the *same* run behind `model_best.npy` (its `training_log.csv`
  is byte-identical to the live one), archived there ahead of starting
  task-4 training, not because it was superseded.

## Unchanged
`agent_code/q_agent/callbacks.py` and `agent_code/q_agent/README.md` — no
diff against `c0f36d7`.
