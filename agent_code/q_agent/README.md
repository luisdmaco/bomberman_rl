# Q-Agent

A linear Q-learning agent for Bomberman: hand-crafted BFS features, action
masking, event-based reward shaping, epsilon-greedy exploration, and online
bootstrapped TD updates stabilised by a target network.

**Final model:** `model.npy` (identical to `old_models/task4/model_round2500.npy`).
Against three `rule_based_agent`s it wins 50% of rounds outright and scores
5.00 points/round vs 2.77 for the best opponent — see [Results](#results).

## Quick start

Run all commands from the repository root.

```bash
# play the final model against three rule_based_agents
python main.py play --my-agent q_agent

# headless evaluation with saved statistics
python main.py play --no-gui --agents q_agent rule_based_agent rule_based_agent rule_based_agent \
  --scenario classic --n-rounds 100 --seed 42 --save-stats results/eval.json

# rank every checkpoint in agent_code/q_agent/ (see evaluate_models.py --help)
python evaluate_models.py --opponents rule_based_agent rule_based_agent rule_based_agent --sort margin

# continue training (first agent in training mode)
python main.py play --no-gui --agents q_agent rule_based_agent rule_based_agent rule_based_agent \
  --train 1 --scenario classic --n-rounds 6000
```

Training requires `model_task3_best.npy` in `agent_code/q_agent/`, because
`setup_training` warm-starts from it. The file currently sits only in
`old_models/task3_better/` and `old_models/task4/`, both gitignored, so
copy it back (or change the path in `setup_training`) before training.

## Files

- `callbacks.py` — `setup`/`act`. Builds the model, loads `model.npy` when not
  training (rejecting checkpoints whose feature count differs from the current
  34), and each step either explores (random *allowed* action, probability
  `epsilon`, training only) or exploits (argmax Q over allowed actions).
- `q_model.py` — `LinearQModel`: one weight vector per action,
  `Q(s, a) = w_a · x`, TD update `w_a += lr * (td_target − w_a · x) * x`, then
  the vector is rescaled if its L2 norm exceeds 10.0. `save`/`load` to `.npy`.
- `features.py` — `state_to_features` (the 34-dim vector) and
  `get_action_mask`, plus helpers:
  - `get_walkable` — tile is enterable (free, no bomb, no other agent)
  - `get_blast_coords` — tiles a bomb at (x, y) hits: `bomb_power` (3) tiles
    in each direction, stopped **only by walls**; crates in the path are
    destroyed but do not block the blast (identical to the game's
    `items.Bomb.get_blast_coords`)
  - `danger_map` / `danger_at` / `danger_feature` — per-tile countdown until a
    blast from current explosions (0) or ticking bombs (timer `t`)
  - `escape_exists` — time-expanded reachability search: can the agent still
    be alive once every relevant bomb has gone off? Accounts for all ticking
    bombs (any owner), burning explosions, the extra step each explosion
    lingers, and optionally the bomb it is about to drop
  - `bfs_direction_and_distance` — first step + distance to the nearest target
  - `bfs_all_distances` — distance + first step to every reachable tile
  - `best_bomb_spot` — first step toward the tile maximising
    `crates_in_blast / (distance + 6)`
  - `safe_free_tiles` — free tiles outside every current/imminent blast
  - `bfs_to_opponents` — first step + distance to the nearest opponent
  - `opponent_in_blast` — would a bomb at (x, y) catch an opponent?
  - `crate_adjacent_free_tiles` — legacy, superseded by `best_bomb_spot`, unused
- `train.py` — `setup_training` / `game_events_occurred` / `end_of_round`:
  hyperparameters, custom events, reward shaping, TD updates, target network,
  checkpointing and the per-round `training_log.csv`.
- `model.npy` — the final weights; overwritten every round during training.
- `training_log.csv` — per-round diagnostics of the last training run (task 4).
- `resul_of_eval_classic.txt` — raw `evaluate_models.py` outputs from every
  evaluation run, tasks 2–4.
- `old_models/` (gitignored) — archived checkpoints: `model_task1_coinheaven.npy`,
  `task2/`, `task3_better/` (incl. `model_task3_best.npy` = its `model_round1000.npy`),
  `task4/` (the final run).

## Features (34)

| Group | Names | Meaning |
|---|---|---|
| walkable (4) | `walkable_{up,down,left,right}` | neighbour tile enterable |
| danger (5) | `danger_here`, `danger_{up,down,left,right}` | `1/(t+1)` blast countdown, 0.0 = safe |
| bomb (1) | `can_bomb` | own bomb available |
| coin dir (5) | `coin_dir_{up,down,left,right,none}` | BFS first step to nearest coin |
| coin dist (1) | `coin_dist` | `1/(d+1)` to nearest coin (1.0 if none reachable) |
| crate dir (5) | `crate_dir_{up,down,left,right,none}` | first step to `best_bomb_spot` |
| safe dir (5) | `safe_dir_{up,down,left,right,none}` | first step to nearest safe tile (only set while in danger) |
| crate blast (1) | `crates_in_blast` | crates a bomb dropped here would destroy, ÷ 12 |
| opponent dir (5) | `opp_dir_{up,down,left,right,none}` | BFS first step to nearest opponent |
| opponent dist (1) | `opp_dist` | `1/d` to nearest opponent (0.0 if none reachable) |
| opponent blast (1) | `opp_in_blast` | a bomb dropped here would catch an opponent |

`crate_dir == NONE` means either "already on the best spot" or "no crates
reachable"; `crates_in_blast` tells the two apart. Same for `opp_dir == NONE`
and `opp_in_blast`.

Because both `get_blast_coords` and the game let blasts pass through crates,
the danger features match the real blast area. They do not underestimate it.

## Actions

`UP`, `DOWN`, `LEFT`, `RIGHT`, `WAIT`, `BOMB`

## Action mask

`get_action_mask` only removes moves that are illegal, provably fatal or
pointless. It never picks the best move; that is left to the learned Q-values.
The mask applies during exploration and exploitation, and to the `max` in the
TD target:

- `WAIT` is banned unconditionally; only the fallback below can re-enable it
- moves into a non-walkable tile, or one that explodes this step (`danger == 0`)
- while any bomb is ticking, moves onto a tile with no escape route
  (`escape_exists`) are pruned, but only while at least one move survives; if
  every move is doomed, only the move whose blast arrives *latest* is kept
- `BOMB` when no bomb is available, the blast would hit neither a crate nor an
  opponent, or `escape_exists` finds no way out after dropping
- fallback: if everything is masked, the walkable neighbour whose blast arrives
  latest is allowed; only if none exists is `WAIT` re-enabled

## Custom events

| Event | Fires when |
|---|---|
| `IN_DANGER` | new tile lies in a current or imminent blast |
| `ESCAPED_DANGER` | was in a blast zone, now out of it |
| `BOMBED_CRATES` | once per crate a just-dropped bomb will hit (immediate credit, no waiting for the fuse) |
| `STALLED` | last 10 positions collapse to ≤ 2 distinct tiles |
| `MOVED_CLOSER_TO_COIN` / `MOVED_FARTHER_FROM_COIN` | BFS distance to nearest coin decreased / increased |
| `MOVED_CLOSER_TO_OPPONENT` / `MOVED_FARTHER_FROM_OPPONENT` | BFS distance to nearest opponent decreased / increased |
| `BOMBED_OPPONENT` | a just-dropped bomb's blast covers at least one opponent |
| `TRAPPED_OPPONENT` | additionally, a covered opponent has no escape route (`escape_exists` from their tile) |
| `WAITED_WITH_OPPONENTS` | chose `WAIT` with opponents on the board (only possible via the mask fallback) |
| `AMBUSH_READY` | bomb available, an opponent in this tile's blast, and dropping would be survivable |
| `WIN_ROUND` | round ends with own score above every opponent's (`end_of_round` only) |

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
| `BOMBED_CRATES` (per crate) | +0.3 |
| `ESCAPED_DANGER` | +0.5 |
| `IN_DANGER` (per step) | −0.15 |
| `MOVED_CLOSER_TO_COIN` / `MOVED_FARTHER_FROM_COIN` | +0.1 / −0.1 |
| `MOVED_CLOSER_TO_OPPONENT` / `MOVED_FARTHER_FROM_OPPONENT` | +0.15 / −0.15 |
| `BOMBED_OPPONENT` | +0.8 |
| `TRAPPED_OPPONENT` | +2.0 (stacks with `BOMBED_OPPONENT`) |
| `WAITED_WITH_OPPONENTS` | −0.1 |
| `AMBUSH_READY` | +0.3 |
| `WIN_ROUND` | +1.5 |
| `STALLED` | −0.3 |
| Every step | −0.03 |

- `GOT_KILLED` and `KILLED_SELF` co-fire on every suicide; `reward_from_events`
  drops `GOT_KILLED` whenever `KILLED_SELF` is present, so a suicide costs −15
  once, not −23.
- `BOMBED_CRATES` is deliberately **linear** in crate count. A super-linear
  bonus made the agent detour for perfect spots and cut bombs/round by 37%.
- Paired closer/farther terms have equal magnitude, so oscillating back and
  forth earns nothing.
- `TRAPPED_OPPONENT` was tuned in isolation from `BOMBED_OPPONENT`; raising
  both together lost in two separate runs.

## Hyperparameters

| | start | decay per round | floor |
|---|---:|---:|---:|
| `epsilon` | 0.2 | ×0.9995 | 0.05 |
| learning rate | 0.005 | ×0.999 | 0.0005 |

- Discount `gamma` = 0.95 (covers the 4-step fuse + explosion; 0.9 did not).
- Target network: `target_model` is resynced from the live weights every 100
  rounds; the TD target bootstraps from it, not from the weights being updated.
- Each action's weight vector is clipped to L2 norm ≤ 10.0 after every update.
- Checkpoint `model_round{N}.npy` every 250 rounds.

## Training & evaluation

TD target per step:
`reward + gamma * max_{a allowed in s'} Q_target(s', a)`; on the terminal
step (`end_of_round`) the target is just the reward.

Each stage continued from the previous stage's best checkpoint: coin-heaven →
loot-crate/classic solo → classic vs `peaceful_agent` + `coin_collector_agent` →
classic vs 3× `rule_based_agent` (warm-started from `model_task3_best.npy`,
6000 rounds). The target network is cloned after the warm-start load, so both
start from the same weights.

`evaluate_models.py` (repo root) copies each `model*.npy` into `model.npy`,
runs a fixed-seed evaluation (`--scenario`, `--n-rounds`, `--seed`,
`--opponents`, up to 3), and ranks the checkpoints by `--sort`
(`score`, `margin`, `coins`, `crates`, `kills`, `suicides`). `margin` is own
score minus the best opponent's, per round. The best checkpoint was usually
from the middle of a run, not the end.

The stats file from `--save-stats` holds only per-agent totals, so the round
win rates below come from 100 separate one-round games (seeds 0–99).

## Results

All evaluations: `classic` unless noted, 100 rounds, seed 42, values per round.

**Task 1** (`coin-heaven`): matched the `coin_collector_agent` BFS baseline:
coins/round gap 0.00, steps/round +0.61.

**Task 2** (solo, 150 rounds, `old_models/task2/model_round4500.npy`), ≈ 92%
of the hand-coded baseline:

| | q_agent | rule_based_agent |
|---|---:|---:|
| coins | 7.89 / 9 | 8.58 |
| crates | 109.9 | 116.6 |
| suicides | 0.000 | 0.000 |
| bombs | 45.9 | 38.1 |
| crates/bomb | 2.40 | 3.07 |

**Task 3** (vs `peaceful_agent` + `coin_collector_agent`,
`model_task3_best.npy`): score 9.78, kills 0.83, suicides 0.29, coins 5.63,
margin **+5.14**. 82.5% of the 3000 training rounds were outright wins.

**Task 4 — final model** (`model.npy`):

| opponents | model | score | best opp. score | kills | suicides | win / tie / loss |
|---|---|---:|---:|---:|---:|---|
| 1× rule_based | task-3 best (baseline) | 6.46 | 3.62 | 0.21 | 0.26 | — |
| 1× rule_based | **final** | 7.03 | 3.09 | 0.22 | 0.19 | **86%** / 0% / 14% |
| 3× rule_based | task-3 best (baseline) | 4.16 | 3.18 | 0.29 | 0.55 | — |
| 3× rule_based | **final** | **5.00** | 2.77 | 0.36 | 0.22 | **50%** / 11% / 39% |

In a 4-player game an agent equal to its opponents would win about 25% of
rounds. Continuing training against `rule_based_agent` mainly cut suicides
(0.55 → 0.22) against three opponents.

Task-4 training curve (6000 rounds, first vs last 250): win rate
44.8% → 63.2%, reward 29.6 → 45.5, kills 0.22 → 0.39, suicides 0.54 → 0.39.

## Development log

**Task 1.** Fixed a broken `np.array` concatenation, the model save/load path
(cwd is the agent dir during callbacks) and an inverted `BOMB` mask condition.
Added epsilon decay and balanced coin-distance shaping to stop oscillation.

**Task 2.**
1. *Agent froze on `WAIT`* — `IN_DANGER: −0.3` outweighed the crate reward, so
   a successful bomb was net-negative. Lowered it and added `BOMBED_CRATES` as
   immediate credit.
2. *Suicided in 97% of rounds* — `gamma = 0.9` was too short-sighted for the
   fuse, and `KILLED_SELF: −5` too cheap. Raised gamma to 0.95, death to −15,
   added `ESCAPED_DANGER`. Suicides → 0.
3. *Training diverged after ~round 2500* — `WAIT` weights blew up under a
   constant learning rate. Added lr decay and 250-round checkpoints.
4. *Bomb placement was the ceiling* — 1.31 crates/bomb vs `rule_based`'s 3.07.
   A `min_crates=3` threshold backfired (bombs −37%). `best_bomb_spot`
   (`crates/(distance+cooldown)`) reached 2.40 crates/bomb with more bombs.

**Task 3.** Checkpoint weight norms showed a Q-learning maximisation-bias
divergence: `WAIT`'s weights kept growing while other actions stayed flat.
Fixed three ways: weight-norm clip, unconditional `WAIT` ban in the mask, and
a target network. Added the opponent features, opponent-directed shaping and a
separate `GOT_KILLED` penalty. `TRAPPED_OPPONENT` had carried reward 0.0 for
two full runs; raising it to +2.0 was the most effective single change.
`escape_exists` was rewritten as a time-expanded search over every ticking
bomb, matching the environment's explosion timing.

**Task 4.** Warm-started from the task-3 best and trained 6000 rounds against
3× `rule_based_agent`; the round-2500 checkpoint was the best on evaluation
and became the final `model.npy`.

## Known limitations

- **`WAIT`'s weights are saturated.** Its weight vector sits at the 10.0 norm
  cap (other actions 2.7–4.2). The maximisation bias is contained rather than
  gone; the mask keeps it harmless by never offering `WAIT` voluntarily.
- **Training log double-counts suicides.** `reward_from_events` drops
  `GOT_KILLED` only from its local copy of the events, so `training_log.csv`
  counts each suicide in both `suicides` and `got_killed`. The rewards
  themselves are correct.
- **Linear model.** Q is linear in the features, so interactions (e.g. "bomb
  only if an opponent is close *and* an escape exists") have to be encoded in
  the features or the mask by hand.
- **Warm-start file.** Training needs `model_task3_best.npy` in the agent
  folder, and the only copies are in gitignored folders; see
  [Quick start](#quick-start).
