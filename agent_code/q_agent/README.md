# Q-Agent

A linear Q-learning agent for Bomberman, using hand-crafted features and epsilon-greedy exploration.

## Files

- `q_model.py` — `LinearQModel`: one linear weight vector per action, updated via TD learning (`w_a += lr * (td_target - w_a·x) * x`). Supports `save`/`load` to/from `.npy`.
- `features.py` — converts a raw game state into the feature vector (`state_to_features`), plus helpers: `get_action_mask` (valid actions), `danger_map` (bomb blast radius), `bfs_direction_and_distance` (shortest path direction/distance to the nearest coin).
- `callbacks.py` — `setup`/`act`: builds the model, and on each step either explores (random allowed action, probability `epsilon`) or exploits (argmax Q-value over allowed actions).
- `train.py` — `setup_training`/`game_events_occurred`/`end_of_round`: hyperparameters, reward shaping, and the TD update loop.
- `model.npy` / `model_task1_coinheaven.npy` — saved weights.

## Features (10)

`walkable_{up,down,left,right}`, `danger_here`, `danger_{up,down,left,right}`, `can_bomb`, `coin_dir_{up,down,left,right,none}`, `coin_dist`

## Actions

`UP`, `DOWN`, `LEFT`, `RIGHT`, `WAIT`, `BOMB`

## Reward shaping

| Event | Reward |
|---|---|
| Coin collected | +1 |
| Killed opponent | +5 |
| Killed self | -5 |
| Invalid action | -1 |
| Bomb dropped | -0.2 |
| In blast danger zone | -0.3 |
| Moved closer to nearest coin | +0.1 |
| Moved farther from nearest coin | -0.1 |
| Every step | -0.1 (encourages fast coin collection) |

## Hyperparameters

- `epsilon`: 0.2 → decays by ×0.997 per round, floor 0.05
- Learning rate: 0.01
- Discount factor (`gamma`): 0.9

## Training

Trained online via bootstrapped TD targets: `reward + gamma * max(Q(next_state, allowed_actions))`. Model weights are saved to `model.npy` at the end of every round.