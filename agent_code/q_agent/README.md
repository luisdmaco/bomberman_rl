# Q-Agent

A linear Q-learning agent for Bomberman, using hand-crafted features, action masking, event-based reward shaping and epsilon-greedy exploration.
(online bootstrapped TD)


## Files

- `q_model.py` — `LinearQModel`: one linear weight vector per action, updated via TD learning (`w_a += lr * (td_target - w_a·x) * x`). Supports `save`/`load` to/from `.npy`.
- `features.py` — converts a raw game state into the feature vector (`state_to_features`), plus helpers:   - `get_walkable` — tile enterable (free, no bomb, no other agent)
  - `get_blast_coords` — tiles a bomb at (x,y) hits (stops at walls, stops after the first crate)
  - `danger_map` / `danger_at` / `danger_feature` — per-tile blast countdown from explosions + ticking bombs
  - `bfs_direction_and_distance` — first-step direction + distance to the nearest target
  - `bfs_all_distances` — distance + first-step direction to *every* reachable tile
  - `best_bomb_spot` — direction to the spot maximising `crates / (distance + cooldown)`
  - `safe_free_tiles` — free tiles not in any current/imminent blast
  - `escape_exists` — can a non-blast tile be reached within the bomb timer (**one bomb only** — see TODO)
  - `get_action_mask` — per-action legality (below)
- `callbacks.py` — `setup`/`act`: builds the model, and on each step either explores (random allowed action, probability `epsilon`) or exploits (argmax Q-value over allowed actions).
- `train.py` — `setup_training`/`game_events_occurred`/`end_of_round`: hyperparameters, custom events, checkpointing, reward shaping, and the TD update loop.
- `model.npy` — current weights, overwritten every round.
- `model_task3_best.npy` — task-3 checkpoint used to warm-start task-4 training (`setup_training` loads it before cloning the target network, so both start in sync instead of from zero).

## Features (34)

| Group | Names | Meaning |
|---|---|---|
| walkable (4) | `walkable_{up,down,left,right}` | neighbour tile enterable |
| danger (5) | `danger_here`, `danger_{up,down,left,right}` | `1/(t+1)` blast countdown, 0.0 = safe |
| bomb (1) | `can_bomb` | a bomb is available |
| coin dir (5) | `coin_dir_{up,down,left,right,none}` | BFS direction to nearest collectable coin |
| coin dist (1) | `coin_dist` | `1/(d+1)` to nearest coin (1.0 if none) |
| crate dir (5) | `crate_dir_{up,down,left,right,none}` | direction to the best bombing spot by `crates/(dist+cooldown)` |
| safe dir (5) | `safe_dir_{up,down,left,right,none}` | direction to nearest safe tile (set only when in danger) |
| crate blast (1) | `crates_in_blast` | crates a bomb dropped here would destroy, ÷ 12 |
| opponent dir (5) | `opp_dir_{up,down,left,right,none}` | BFS direction to nearest opponent |
| opponent dist (1) | `opp_dist` | `1/d` to nearest opponent (0.0 if none reachable) |
| opponent blast (1) | `opp_in_blast` | would a bomb dropped here catch an opponent? |

`crate_dir == NONE` means either "already on the best spot" or "no crates
reachable"; `crates_in_blast` disambiguates. Same idea for `opp_dir == NONE`
vs `opp_in_blast`.

## Actions

`UP`, `DOWN`, `LEFT`, `RIGHT`, `WAIT`, `BOMB`


## Action mask

Applied before exploration and exploitation, and to the TD bootstrap `max`:

- `WAIT` is banned unconditionally, regardless of solo/multiplayer — only the
  fallback below can re-enable it, as a genuine last resort
- move into a non-walkable tile, or one that explodes this step (`danger == 0`)
- while any bomb is ticking: moves into a tile with no escape route are pruned,
  but only while at least one other move still survives; if every move is
  doomed, nothing is pruned there — instead only the move whose blast arrives
  *latest* is kept, so the agent runs rather than freezing
- `BOMB` when: no bomb available; the blast would hit neither a crate nor an
  opponent; or `escape_exists` says the current tile has no way out
- fallback: if every action is masked, move to whichever walkable neighbour is
  least imminently dangerous; only if none exist does `WAIT` get re-enabled


## Custom events

| Event | Fires when |
|---|---|
| `IN_DANGER` | new tile is inside a blast zone |
| `ESCAPED_DANGER` | was in a blast zone, now out of it |
| `BOMBED_CRATES` | once per crate a just-dropped bomb will hit (immediate credit, no waiting for the fuse) |
| `STALLED` | last 10 positions collapse to ≤ 2 distinct tiles |
| `MOVED_CLOSER_TO_COIN` / `MOVED_FARTHER_FROM_COIN` | BFS distance to nearest coin decreased / increased |
| `MOVED_CLOSER_TO_OPPONENT` / `MOVED_FARTHER_FROM_OPPONENT` | BFS distance to nearest opponent decreased / increased |
| `BOMBED_OPPONENT` | a just-dropped bomb's blast catches at least one opponent |
| `TRAPPED_OPPONENT` | a caught opponent additionally has no escape route (checked the same way the mask checks our own) |
| `WAITED_WITH_OPPONENTS` | chose `WAIT` while an opponent is on the board — belt-and-suspenders alongside the mask's unconditional ban |
| `AMBUSH_READY` | bomb available, an opponent already sits in this tile's blast, and dropping now wouldn't be suicide — fires without a bomb actually being dropped |
| `WIN_ROUND` | round ends with our score above every opponent's (`end_of_round` only, training-only signal) |

## Reward shaping

| Event | Reward |
|---|---:|
| Coin collected | +1 |
| Killed opponent | +6 |
| Killed self | −15 |
| Got killed (by someone else) | −8 |
| Invalid action | −1 |
| Bomb dropped | 0.0 |
| Crate destroyed | +0.5 |
| Coin found | +0.3 |
| Survived round | +0.5 |
| `BOMBED_CRATES` (per crate, at drop time) | +0.3 |
| `ESCAPED_DANGER` | +0.5 |
| `IN_DANGER` (per step in a blast zone) | −0.15 |
| `MOVED_CLOSER_TO_COIN` / `MOVED_FARTHER_FROM_COIN` | +0.1 / −0.1 |
| `MOVED_CLOSER_TO_OPPONENT` / `MOVED_FARTHER_FROM_OPPONENT` | +0.15 / −0.15 |
| `BOMBED_OPPONENT` | +0.8 |
| `TRAPPED_OPPONENT` | +2.0 (stacks with `BOMBED_OPPONENT`) |
| `WAITED_WITH_OPPONENTS` | −0.1 |
| `AMBUSH_READY` | +0.3 |
| `WIN_ROUND` | +1.5 |
| `STALLED` | −0.3 |
| Every step | −0.03 |

`GOT_KILLED` and `KILLED_SELF` co-fire on every suicide; `reward_from_events`
drops `GOT_KILLED` from the tally whenever `KILLED_SELF` is also present, so a
suicide is priced once at −15, not −23.

`BOMBED_CRATES` is deliberately **linear** in crate count: each bomb costs the
same ~10 steps, and the step penalty already prices travel time. A super-linear
bonus was tried and caused the agent to detour for perfect spots, cutting bombs
per round by 37%.

`TRAPPED_OPPONENT` and `BOMBED_OPPONENT` are kept separate rather than raised
together — an earlier attempt raising both at once lost twice in a row,
confounding which change helped.


## Hyperparameters

- `epsilon`: 0.2 → decays by ×0.9995 per round, floor 0.05
- Learning rate: 0.005 → decays by ×0.999 per round, floor 0.0005
- Discount factor (`gamma`): 0.95
- Target network: weights snapshotted into `target_model` every 100 rounds
  (`target_sync_every`); the TD bootstrap always reads from this stale copy,
  not the live weights, to stop `max`-bootstrap self-reinforcement
- Weight vectors are hard-clipped to L2 norm 10.0 after every update, as a
  second, independent guard against the same divergence

## Training & evaluation

Trained online via bootstrapped TD targets: `reward + gamma * max(Q(next_state, allowed_actions))`. Model weights are saved to `model.npy` at the end of every round.

Task-4 training warm-starts from `model_task3_best.npy` (loaded in
`setup_training`, before the target network is cloned, so both start in sync)
instead of from zero-initialized weights.

`evaluate_models.py` (repo root) swaps each checkpoint into `model.npy`, runs a
fixed-seed eval, and ranks them. The best model is often **mid-run**, not the endpoint.


## Results

Task 1 (`coin-heaven`, 100 rounds): matched the `coin_collector_agent` BFS
baseline — coins/round gap 0.00, steps/round +0.61.

Task 2 (`classic` solo, 150 rounds, seed 42, `model_round4500.npy`):

| | q_agent | rule_based_agent |
|---|---|---|
| coins | **7.89** / 9 | 8.58 |
| crates | 109.9 | 116.6 |
| suicides | **0.000** | 0.000 |
| bombs | 45.9 | 38.1 |
| crates/bomb | 2.40 | 3.07 |

≈ 92% of the hand-coded baseline. Checkpoints from rounds 2000–5500 all score
7.09–7.89, i.e. training is stable rather than diverging.

Task 3 (`classic` vs `peaceful_agent` + `coin_collector_agent`, 100 rounds,
seed 42, `model_round1000.npy`): score 9.78, kills 0.83, suicides 0.29,
coins 5.63, crates 63.61, margin **+5.14**. Across the full 3000-round training
run behind this checkpoint, 82.5% of rounds ended in an outright win.

Task 4, in progress (`classic` solo vs one `rule_based_agent`, 100 rounds,
seed 42): best checkpoint so far `model_round2500.npy` — score 7.03,
kills 0.22, suicides 0.19, coins 5.93, crates 69.42, margin **+3.94**, clearly
ahead of the un-continued `model_task3_best.npy` itself (margin +2.55).
Training curve of the 3×`rule_based_agent` run (first vs last 250 rounds):
win rate 44.8% → 63.2%, kills 0.22 → 0.39, suicides 0.54 → 0.39 — improving
cleanly; full 3-opponent evaluation not yet run to completion.

## Development log

**Task 1.** Fixed a broken `np.array` concatenation, the model save/load path
(cwd is the agent dir during callbacks), and an inverted `BOMB` mask condition.
Added epsilon decay and balanced coin-distance shaping to stop oscillation.

**Task 2.**
1. *Agent froze on `WAIT`* — `IN_DANGER: −0.3` outweighed the crate reward, so a
   successful bomb was net-negative. Fixed to −0.05 + immediate `BOMBED_CRATES`.
2. *Suicided in 97% of rounds* — `gamma = 0.9` (~10-step horizon) was shorter
   than the fuse, so survival was invisible; `KILLED_SELF: −5` was cheap against
   a ~+26 round. Raised gamma to 0.95, death to −15, added `ESCAPED_DANGER`,
   masked `WAIT` in blast zones. Suicides → 0.
3. *Training diverged after ~round 2500* — `WAIT` weights blew up ~10× under a
   constant lr with higher gamma. Added lr decay + 250-round checkpoints.
4. *Bomb placement was the ceiling* — benchmarking showed `rule_based` drops the
   same number of bombs at 3.07 crates each vs our 1.31. A `min_crates=3`
   threshold backfired (yield +7%, bombs −37% from detours). `best_bomb_spot`,
   scoring by `crates/(distance+cooldown)`, fixed it: 2.40 crates/bomb with
   bombs *up* to 45.9.

**Task 3.** Diagnosed a Q-learning maximization-bias divergence via checkpoint
weight norms (`WAIT`'s vector growing unbounded while every other action
stayed flat). Fixed three ways: hard weight-norm clip in `q_model.py`,
banning voluntary `WAIT` outright in the mask, and a target network in
`train.py` (bootstrap off a 100-round-stale weight copy). Added opponent
features (`opp_dir`, `opp_dist`, `opp_in_blast`), opponent-directed reward
shaping, and gave `GOT_KILLED` its own penalty (previously unrewarded because
it always co-fired with `KILLED_SELF` in solo play). `TRAPPED_OPPONENT` had
sat at reward 0.0 for two full runs despite firing regularly — raising it to
+2.0 was the single highest-leverage change. Result: 82.5% round win rate
against `peaceful_agent` + `coin_collector_agent`.

**Task 4 (in progress).** Warm-starting from `model_task3_best.npy` instead of
retraining from scratch, against tougher `rule_based_agent` opponents.

## TODO — task 4

### Resolved from the task-3 TODO
- `escape_exists` now accounts for every ticking bomb in `game_state['bombs']`
  regardless of owner, not just a hypothetical own bomb — safe with multiple
  simultaneous opponent bombs.
- `GOT_KILLED` has its own reward (−8), distinct from `KILLED_SELF` (−15).
- Opponent features (`opp_dir`, `opp_dist`, `opp_in_blast`) are in place — 34
  features total.

### Still open
- **CSV logging double-counts suicides.** `reward_from_events` filters
  `GOT_KILLED` out of its own *local* copy of `events` before summing the
  reward, but `game_events_occurred`/`end_of_round` still call
  `note_events(self, events, reward)` with the original, unfiltered list — so
  `training_log.csv` counts a suicide under both `suicides` and `got_killed`.
  The reward math is correct; only the diagnostic CSV column is off.
- **No second, structurally different model yet.** The project asks for two
  or more distinct techniques; only the linear Q-learning agent exists so far.
- Full task-4 evaluation against 3 simultaneous `rule_based_agent`s hasn't
  been run to completion yet.