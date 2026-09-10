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

## Features (27)

| Group | Names | Meaning |
|---|---|---|
| walkable (4) | `walkable_{up,down,left,right}` | neighbour tile enterable |
| danger (5) | `danger_here`, `danger_{up,down,left,right}` | `1/(t+1)` blast countdown, 0.0 = safe |
| bomb (1) | `can_bomb` | a bomb is available |
| coin dir (5) | `coin_dir_{up,down,left,right,none}` | BFS direction to nearest collectable coin |
| coin dist (1) | `coin_dist` | `1/(d+1)` to nearest coin (1.0 if none) |
| crate dir (5) | `crate_dir_{up,down,left,right,none}` | direction to the best bombing spot by `crates/(dist+cooldown)` |
| safe dir (5) | `safe_dir_{up,down,left,right,none}` | direction to nearest safe tile (set only when in danger) |
| crate blast (1) | `crates_in_blast` | crates a bomb dropped here would destroy, ÷ 4 |


`crate_dir == NONE` means either "already on the best spot" or "no crates
reachable"; `crates_in_blast` disambiguates.

## Actions

`UP`, `DOWN`, `LEFT`, `RIGHT`, `WAIT`, `BOMB`


## Action mask

Applied before exploration and exploitation, and to the TD bootstrap `max`:

- move into a non-walkable tile, or one that explodes this step (`danger == 0`)
- `WAIT` inside a blast zone, or whenever playing solo (`len(others) == 0`) —
  nothing changes while waiting, and a WAIT-favouring Q locks the agent in place
- `BOMB` when: no bomb available; no crates and no opponents; solo with zero
  crates in blast; or `escape_exists` is false
- fallback: if every action is masked, `WAIT` is re-enabled


## Custom events

| Event | Fires when |
|---|---|
| `IN_DANGER` | new tile is inside a blast zone |
| `ESCAPED_DANGER` | was in a blast zone, now out of it |
| `BOMBED_CRATES` | once per crate a just-dropped bomb will hit (immediate credit, no waiting for the fuse) |
| `STALLED` | last 10 positions collapse to ≤ 2 distinct tiles |
| `MOVED_CLOSER_TO_COIN` / `MOVED_FARTHER_FROM_COIN` | BFS distance to nearest coin decreased / increased |

## Reward shaping

| Event | Reward |
|---|---|
| Coin collected | +1 |
| Killed opponent | +5 |
| Killed self | -5 |
| Invalid action | -1 |
| Bomb dropped | -0.2 |
| Crate destroyed | +0.5 |
| Coin found | +0.3 |
| Survived round | +0.5 |
| `BOMBED_CRATES` (per crate, at drop time) | +0.3 |
| `ESCAPED_DANGER` | +0.4 |
| `IN_DANGER` (per step in a blast zone) | −0.05 |
| `MOVED_CLOSER_TO_COIN` / `MOVED_FARTHER_FROM_COIN` | +0.1 / −0.1 |
| `STALLED` | −0.3 |
| Every step | −0.02 |

`BOMBED_CRATES` is deliberately **linear** in crate count: each bomb costs the
same ~10 steps, and `STEP_PENALTY` already prices travel time. A super-linear
bonus was tried and caused the agent to detour for perfect spots, cutting bombs
per round by 37%.


## Hyperparameters

- `epsilon`: 0.2 → decays by ×0.995 per round, floor 0.05
- Learning rate: 0.01 → decays by ×0.995 per round, floor 0.001
- Discount factor (`gamma`): 0.95

## Training & evaluation

Trained online via bootstrapped TD targets: `reward + gamma * max(Q(next_state, allowed_actions))`. Model weights are saved to `model.npy` at the end of every round.

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

## TODO — task 3 / 4 (opponents)

The agent has never seen an opponent. The tournament is 4-player and a kill is
worth 5 points (> 5 coins), so this is the critical gap.

### Must fix first
- **`escape_exists` is broken with opponents.** It only accounts for the
  hypothetical own bomb and ignores every enemy bomb, so the mask will approve
  bombs whose escape route runs through someone else's blast. Safe in solo (only
  one bomb possible); unsafe with up to 4 agents.
- **`GOT_KILLED` has no reward.** It was skipped in task 2 because it always
  co-fired with `KILLED_SELF`; with opponents it becomes a distinct event and
  needs its own penalty.

### New features needed
- `opp_dir_{up,down,left,right,none}` — direction to nearest opponent
- `opp_dist` — normalised distance
- `opp_in_blast` — would a bomb here catch an opponent?

(+7 → 34 features; invalidates existing checkpoints, so retrain from scratch.)

### Already opponent-ready
`danger_map`, `safe_free_tiles`, and `safe_dir` handle multiple bombs correctly.
The `WAIT` and zero-crate-bomb masks are guarded by `len(others) == 0`, so both
re-enable themselves automatically in multi-agent play.