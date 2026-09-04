# DQN arm: working log

Owner: Luis. Everything about the deep Q-network agent lives here.
Read Part 1 once to know what you are looking at. Part 2 is the running history,
newest first.

---

## STATUS (2026-09-04)

Gate 1 is **not** passed. The agent is currently worse than a random walk.

| | Coins per round | What it means |
|---|---|---|
| `rule_based_agent` | 50.0 | The ceiling. Collects everything in ~125 steps. |
| **Gate 1 target** | **45** | What we need. |
| Random walk | ~18 | The floor. Wandering blindly still finds coins. |
| Our frozen DQN | ~4 to 5 | Where we actually are. |
| Our DQN, measured wrong | 45 | An illusion. See entry 2026-09-03. |

Two tooling bugs found and fixed so far. Neither was a modelling problem, both
were measurement or plumbing problems, and both made the numbers lie.

3000 clean rounds produced no learning: frozen coins wandered between 2.6 and
6.2 with no trend, and `mean_steps` was exactly 400.0 at every checkpoint, so
the board was never cleared. `invalid_per_round` did fall to ~0, so the network
was learning wall avoidance. Cause found: the reward function. See the
2026-09-05 entry.

Next run to do, now that the reward is fixed:

```
python tools/sweep.py --rounds 3000 --jobs 6
```

Six configs in parallel, one core and ~100 MB each. The decisive comparison is
`baseline` (fixed shaping) vs `no_shaping` vs `old_shaping` (the bug, kept
deliberately for the report's ablation).

---

# Part 1 — How this works

## The task

Scenario `coin-heaven`: a 17x17 board, no crates, no opponents, 50 coins all
visible from step 1, 400 steps to collect them. Pure navigation. No bombs
needed, so we mask the BOMB action out during this stage.

## The pipeline

There are two completely separate things the code does, and confusing them is
what caused our biggest bug.

**Training.** `main.py play --train 1` runs the game and calls into
`train.py` after every step. The agent plays, stores what happened, and updates
its network weights every 4 steps. Its weights are changing constantly.

**Evaluation.** `main.py play` with no `--train` loads the saved weights and
plays with them **frozen**. Nothing updates. This is the only mode that
describes the agent we would actually submit, because in the tournament nothing
learns.

Those two can disagree wildly. Ours disagreed by a factor of 7.

## What each file is

| File | What it is | Trust it? |
|---|---|---|
| `agent_code/dqn_agent/training_log.csv` | One row per round **during** training | For debugging only. NOT a learning curve. |
| `agent_code/dqn_agent/eval_curve.csv` | Frozen model measured at each checkpoint | **Yes. This is the real curve.** |
| `agent_code/dqn_agent/checkpoints/dqn-rNNNNNN.pt` | Weights snapshot at round NNNNNN | The raw material for the curve |
| `agent_code/dqn_agent/dqn-model.pt` | Latest weights | What gets submitted |
| `agent_code/dqn_agent/logs/dqn_agent.log` | Every action and its Q values | For diagnosing weird behaviour |

## Reading `eval_curve.csv`

This is the one that matters.

| Column | Meaning | What good looks like |
|---|---|---|
| `rounds_trained` | How much training had happened | x-axis of your plot |
| `frozen_coins` | Coins per round, training OFF | Must climb past 18, then toward 45 |
| `ci95` | Half-width of the 95% confidence interval | Smaller is more certain. If two points overlap, they are not different. |
| `mean_steps` | Steps per round | **400 means it never finished.** Dropping toward ~150 means it is collecting everything and ending early. This is a great early signal. |
| `invalid_per_round` | Times it walked into a wall | Should go to ~0 |

A flat `frozen_coins` means it is not learning. A `mean_steps` stuck at exactly
400.0 means it is never clearing the board.

## Reading `training_log.csv`

Useful for debugging, misleading as a result.

`round, steps, score, coins, invalid_actions, waited, reward_sum, epsilon, mean_loss, buffer`

The two columns to actually check here:

- `epsilon` should fall smoothly from 1.0 to 0.05 across the whole run. If it
  jumps back up, something restarted. That was bug #2.
- `buffer` should climb to 50000 and stay. If it drops back to ~400, the replay
  memory got wiped. Same bug.

## The commands

Train and build the honest curve, one continuous run:

```
DQN_ACTIONS="UP,RIGHT,DOWN,LEFT" DQN_VIEW="ego:13" DQN_EPS_DECAY=1500 \
  python tools/train_curve.py --rounds 3000 --checkpoint-every 250 --eval-rounds 40
```

Rebuild the curve from checkpoints already on disk, no retraining:

```
python tools/train_curve.py --eval-only
```

Measure one agent against the gate:

```
python tools/evaluate.py --agents dqn_agent --scenario coin-heaven --n-rounds 200 --gate 1
```

Set `DQN_EPS_DECAY` to roughly half your total rounds, so exploration anneals
over the first half and the second half exploits.

---

# Part 2 — Log

## 2026-09-05 — Bug #3: the reward function paid the agent to do nothing

**Symptom.** 3000 rounds of clean continuous training, frozen curve flat:
2.90, 3.27, 2.58, 4.58, 4.78, 6.17, 3.50, 4.58, 3.92, 4.28, 3.60, 4.97.
`mean_steps` 400.0 at every checkpoint. `invalid_per_round` near 0.

That combination is the tell: it HAD learned (wall avoidance is real learning),
it just had not learned to collect coins. So the problem was what we were paying
it for, not whether it could learn.

**Cause.** The shaping term `F = gamma * Phi(s') - Phi(s)` with `Phi = -0.1*d`
and gamma 0.95. Two consequences, both arithmetic, both verifiable by hand:

1. When the distance does not change, F leaves a residual of
   `Phi(s)*(gamma-1) = +0.005*d` per step. Shuffling between two tiles 10 steps
   from the nearest coin paid **+0.065 every two steps, forever**. At distance
   15, +0.115. Over a 400-step round that is roughly +13 to +23 for making no
   progress whatsoever.
2. Collecting a coin makes the nearest-coin distance jump, so Phi drops
   discontinuously. Once the next coin was 12 or more steps away, collecting was
   **net negative**: -0.06 at distance 12, -0.35 at distance 15, and as bad as
   -1.77 in the worst case against a +1.0 coin reward.

So late in a round, when coins get sparse, the reward function punished the
agent for finishing and paid it to hover. It learned exactly that. This is
precisely the bad local optimum the project spec warns about, and it was my
reward design, not the network.

**Fix.** Three parameters, all now separately controllable:

| Parameter | Was | Now | Why |
|---|---|---|---|
| `DQN_SHAPING_GAMMA` | tied to gamma, 0.95 | 1.0 | Removes the per-step drift entirely. Standing still is worth exactly 0, so the step penalty makes hovering strictly negative. |
| `DQN_SHAPING_SCALE` | 0.1 | 0.05 | Keeps collecting net positive. |
| `DQN_SHAPING_CAP` | none | 10 | Flattens the potential past 10 tiles, bounding the jump when a coin is collected. |

Verified all three behaviours now have the right sign:

| Behaviour | Before | After | Required |
|---|---|---|---|
| Oscillate at d=10, per 2 steps | +0.065 | -0.040 | negative |
| Step toward a coin | +0.100 | +0.030 | positive |
| Collect a coin, worst case | -1.770 | +0.530 | positive |

Setting the shaping discount to 1.0 gives up the exact policy-invariance of
Ng, Harada & Russell (1999), which holds for gamma_shaping = gamma. That is a
deliberate trade: the theorem guarantees the asymptotic optimum is unchanged,
but says nothing about a learner with a 20-step effective horizon finding it,
and the drift version handed that learner a dense reward for standing still.

**Not yet known.** Whether the fixed reward actually learns. A 200-round smoke
test still scored 4, but epsilon had barely annealed and learning had only just
started, so it says nothing either way. The sweep is the real test.

**For the report.** `old_shaping` is kept as a sweep config so the broken
version can be measured against the fixed one. A shaping design that provably
creates a hovering optimum, with the arithmetic and the measured flat curve, is
a genuine result.


## 2026-09-04 — Made training cheaper, and parallel

**Question asked.** Can training go faster on an M3 Max (36 GB) without hitting swap?

**Measured first.** `tools/profile_step.py` breaks one round down:

| Cost per 400-step round | Share |
|---|---|
| Gradient steps | **92%** |
| Network forward pass (acting) | 5% |
| `state_to_features` | 2% |
| BFS shaping | 0.2% |

So the game loop is irrelevant and the gradient step is everything. Also
measured: a second torch thread buys under 10% on a network this small, and
batch 32 is the least efficient batch size per sample by a wide margin.

**Machine was never the constraint.** 27.45 of 36 GB used, **0 bytes of swap**,
CPU 76% idle. A single training run cannot use more than about one core because
the game loop is serial Python.

**Changes.**
1. Replay buffer stores states as `uint8` instead of `float32`. Five of the six
   channels are binary and the sixth is a 0..1 danger level, so the precision
   was never used. 50,000 transitions: **406 MB -> 101 MB per process.**
2. Default batch 32 every 4 steps -> **batch 128 every 16 steps**. Identical
   replay ratio (8 samples per environment step, so sample efficiency is
   unchanged) but a quarter as many per-update overheads.
3. `DQN_THREADS` added, default 1. One thread per process, many processes.
4. `tools/sweep.py`: runs several configurations at once, each in its own agent
   directory so checkpoints and logs never collide. Default configs are the
   ablations the report needs: shaping on/off, egocentric vs global view, two
   learning rates, and old-vs-new batch settings.

The spec forbids multiprocessing in the *submitted agent* but explicitly allows
it for training.

**What this does not do.** None of it makes the agent better. It means the whole
ablation runs in the time one run used to take, which is how we find out what
actually fixes the frozen-policy problem.


## 2026-09-04 — Bug #2: chunked training reset everything

**Symptom.** Frozen curve flat and drifting down over 1500 rounds:
5.5, 4.75, 4.25, 4.88, 4.15, 3.98.

**Cause.** `tools/train_eval_curve.py` trained in chunks by re-invoking
`main.py`. Each invocation is a fresh Python process, so per chunk:

- epsilon restarted at 1.0 and only reached 0.70 by the chunk's end
- the 50,000-transition replay buffer was wiped and refilled from scratch
- the Adam optimiser state was thrown away

Only the network weights carried over. The agent trained for 1500 rounds at
70 to 100 percent random actions, six times over, with amnesia in between.

**Evidence.** `training_log.csv` showed `round 1, eps 0.9988, buffer 400` and
`round 250, eps 0.7031, buffer 50000` repeating identically for every chunk.

**Fix.** Replaced with `tools/train_curve.py`: ONE continuous training process,
so epsilon, buffer and optimiser state are all continuous. It writes numbered
checkpoints as it goes, then a second pass replays each one frozen to build the
curve. Verified: epsilon now reaches 0.05 and the buffer grows to 48000 without
resetting.

**Not yet known.** Whether continuous training actually reaches 45 coins.
A 120-round smoke test still scored 2 to 5, but that is far too early to judge.

## 2026-09-03 — Bug #1: the training log was measuring the wrong agent

**Symptom.** `training_log.csv` reported 45 of 50 coins by round 1500.
Evaluating the same saved weights with training off gave **6.2**.

**Diagnosis.** Traced one frozen round: the agent walked ~25 steps, then chose
LEFT, RIGHT, LEFT, RIGHT for the remaining 380 steps. 192 of each. A
deterministic policy on a board that has not changed is a fixed point, so if the
argmax says "right" at A and "left" at B it shuttles forever.

**Proof.** Resumed from the identical saved file. Learning OFF: stayed at ~7
coins. Learning ON: 7, 12, 16, 33, 43, **45** within ten rounds. The weight
drift during training was what kept breaking the loops, not skill.

**Also ruled out.** Play-time randomness barely helped (6.2 to 9.5 at eps 0.10)
and cost invalid actions, so the loop was a symptom, not the cause.

**Fixes applied.**
1. Egocentric view (`DQN_VIEW=ego:13`): a 13x13 window recentred on the agent,
   padded with walls. The global 17x17 view forced the network to learn "walk
   toward a coin" separately for every square on the map.
2. Timeout is now treated as truncation, not termination (Pardo et al. 2018).
   Storing step 400 with `done=True` taught the agent the world ends there, and
   handed it a shaping bonus for being far from a coin when the clock stopped.

**For the report.** Both of these are ablatable: `DQN_VIEW=global` restores the
old view. The 7x online-versus-frozen gap is itself a result worth writing up,
and it applies to the Q-learning arm too.

## 2026-09-03 — Bug #0: the agent kept bombing itself

**Symptom.** Training rounds ended after 5 to 12 steps instead of 400.

**Cause.** With epsilon near 1.0 the agent picked BOMB at random, killed itself,
and `--train` ends a round when the last training agent dies.

**Fix.** `DQN_ACTIONS="UP,RIGHT,DOWN,LEFT"` masks BOMB and WAIT for this stage.
The spec's own definition of task 1 says it "does not require dropping any
bombs", so this is a documented curriculum choice. The network keeps all six
outputs, so weights transfer to tasks 2 to 4. **Remove the mask for task 2.**

## 2026-09-03 — Agent built

Double DQN, target network, uniform replay, potential-based reward shaping.
353k parameters, of which only 6k are convolutional and 347k are the
fully-connected head. Six input channels: walls, crates, self, others, coins,
danger. Every hyperparameter reads from an environment variable so sweeps need
no source edits. See `agent_code/dqn_agent/README.md` for the full table.

Baselines measured the same day, 30 rounds each on coin-heaven:
`rule_based_agent` 50.0 coins in 125.2 steps with 0 invalid actions; untrained
random walk ~18 coins with ~145 invalid actions.

## 2026-08-31 — Project plan

Team of 3. Two models: feature-based Q-learning (ships to the tournament) and
this DQN (comparison, explicitly allowed to fail). Four-gate curriculum. Code
due 21.09, report 28.09, free crash test 17.09. See `PROJECT_PLAN.md`.

---

## Open questions

- Does continuous training actually reach 45 coins, or is the architecture the
  limit? Unknown until a 3000-round run finishes.
- The `danger` channel has never been exercised. It is all zeros in coin-heaven.
  Re-verify the bomb-timer indexing against `environment.do_step` before task 2.
- Uniform replay only. Prioritised replay is a candidate upgrade with a baseline
  to compare against.
