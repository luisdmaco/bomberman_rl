# dqn_agent

Deep Q-Network arm of the project. Double DQN with a target network and uniform
experience replay, over a convolutional encoder that reads the raw board.

The point of this agent is that it learns its own features, so that the report
can compare learned features against the engineered features used by the
Q-learning agent. It deliberately contains no pathfinding, no danger heuristics
and no hand-written rules in `state_to_features`.

## Files

| File | Contents |
|---|---|
| `callbacks.py` | `setup`, `act`, `state_to_features`. The only file loaded in official games. |
| `train.py` | `setup_training`, `game_events_occurred`, `end_of_round`, rewards, the learning step. Never imported when `train=False`. |
| `model.py` | Network, replay buffer, action masking. |
| `dqn-model.pt` | Saved weights. Written every `DQN_SAVE_EVERY` rounds. |
| `training_log.csv` | One row per round. This is what the report's training curves are plotted from. |

## State encoding

Six channels of the full board, each `COLS x ROWS`:

`walls`, `crates`, `self`, `others`, `coins`, `danger`

Channels 1, 3 and 5 are all-zero in coin-heaven. They exist from the start so
the same architecture carries into tasks 2 to 4 without retraining from scratch.

## Training

Task 1, coin-heaven:

```
DQN_ACTIONS="UP,RIGHT,DOWN,LEFT" DQN_EPS_DECAY=600 \
  python main.py play --no-gui --agents dqn_agent --train 1 \
  --scenario coin-heaven --n-rounds 1500
```

`DQN_ACTIONS` restricts the action space for this curriculum stage. Without it
the exploring agent drops bombs, kills itself within a few steps, and the round
ends before it can learn anything: `--train` stops a round as soon as the last
training agent dies. The spec's own definition of task 1 says it "does not
require dropping any bombs", so restricting the set here is a stage choice, not
a shortcut. **Remove the mask for tasks 2 to 4.** The network always has all six
outputs, so the weights transfer.

Resume an interrupted run with `DQN_RESUME=1`.

## Evaluation

```
python tools/evaluate.py --agents dqn_agent --scenario coin-heaven \
  --n-rounds 200 --gate 1
```

Runs with training off, which is how the tournament and the official submission
test run the agent. Reference points on coin-heaven, 30 rounds:

| Agent | Coins / round | Steps / round | Invalid / round |
|---|---|---|---|
| `rule_based_agent` | 50.00 | 125.2 | 0.00 |
| untrained network (random walk) | ~18 | 401 | ~145 |

## Hyperparameters

Every value in `CONFIG` reads from the environment, so a sweep is a shell loop
rather than a source edit, and every run is reproducible from its command line.

| Variable | Default | Notes |
|---|---|---|
| `DQN_GAMMA` | 0.95 | Short horizon suits coin chasing. Raise for tasks 3 and 4. |
| `DQN_LR` | 5e-4 | Adam. |
| `DQN_BATCH` | 32 | |
| `DQN_BUFFER` | 50000 | About the last 125 rounds. |
| `DQN_LEARN_START` | 2000 | Transitions collected before the first gradient step. |
| `DQN_TRAIN_EVERY` | 4 | Environment steps per gradient step. |
| `DQN_TARGET_UPDATE` | 1000 | Gradient steps between target syncs. |
| `DQN_EPS_START` / `DQN_EPS_END` | 1.0 / 0.05 | |
| `DQN_EPS_DECAY` | 400 | Rounds over which epsilon decays linearly. |
| `DQN_SHAPING` | 1 | Set to 0 for the ablation. |
| `DQN_SHAPING_SCALE` | 0.1 | Potential is `-scale * BFS distance to nearest coin`. |
| `DQN_SEED` | -1 | Set a value for reproducible runs. |
| `DQN_REWARD_PROFILE` | `curriculum` | `score` keeps only scoreboard progress plus safety shaping. `score_hunt` preserves that profile and adds +0.5 when an escapable newly dropped bomb currently threatens an opponent; crate-only bombs remain neutral. |

## Reward design

Sparse game rewards plus a step penalty, plus optional potential-based shaping.

The shaping term is `gamma * Phi(s') - Phi(s)` with `Phi(s) = -0.1 * d(s)`,
where `d` is the BFS distance to the nearest reachable coin. This is the form
from Ng, Harada and Russell (1999), which the project spec cites in a footnote:
a shaping term of this shape provably leaves the optimal policy unchanged. That
is why the potential depends only on the state and never on the action that
reached it. The BFS runs only during training, so it costs nothing against the
0.5 s per-step budget in official games.

`DQN_SHAPING=0` gives the ablation for the report: same everything, no shaping.

For the targeted tournament fine-tune, `DQN_REWARD_PROFILE=score_hunt` adds an
immediate +0.5 signal when an escapable bomb has an opponent in its blast line.
The eventual kill remains worth +5.0. Suicidal bombs never receive the targeted
bonus, and ordinary crate demolition remains worth zero under this profile.
The original `score` profile is unchanged so the r1250 result stays exactly
reproducible.

## Known gaps

- **The `danger` channel is not yet verified.** It is all zeros in coin-heaven,
  so gate 1 does not exercise it. The exact bomb-timer indexing must be checked
  against `environment.do_step` before task 2 depends on it.
- **Uniform replay, not prioritised.** Prioritised replay is a planned upgrade;
  uniform first so the upgrade has an honest baseline to be measured against.
- **`torch` must be in `requirements.txt`** if this agent is the one submitted.
