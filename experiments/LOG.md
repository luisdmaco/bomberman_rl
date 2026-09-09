# DQN arm: working log

Owner: Luis. Everything about the deep Q-network agent lives here.
Read Part 1 once to know what you are looking at. Part 2 is the running history,
newest first.

---

## STATUS (2026-09-09, three task-2 runs in)

Task 2 best agent: **run 1 @ r3000 played with the suicide mask, 6.94 coins and
5.67% suicide** over 300 frozen rounds. Gate is 8 coins and 2%. Both still
climbing steeply when the runs ended, so the next thing is simply a longer run.
12 days to the code deadline, 8 to the crash test.

| | |
|---|---|
| Task 1 best (frozen) | 47.39 coins in 234 steps |
| Task 2 gate | suicide rate < 2%, 8 of 9 coins |
| Task 2 best (run1 r3000 + mask, 300 rounds) | **6.94 coins**, **5.67% suicide**, 0 invalid |
| ... in rounds where it survives | 7.27 coins, 52% reach 8 or more |

**The one thing that has changed conceptually:** the suicide mask is now
play-time only. Training with it on measurably hurts. See the third
2026-09-09 entry.

Weights are archived as `models/task2-run1-r3000-ego13.pt`. Run 3 overwrote
`checkpoints/` and `dqn-model.pt`, so that archive is the only copy.

**Run this next.** Longer, and the mask now switches itself off for training:

```
DQN_VIEW="ego:13" DQN_EPS_DECAY=3000 python tools/train_curve.py \
  --rounds 6000 --checkpoint-every 250 --eval-rounds 40 --scenario crate-light
```

Roughly 90 minutes. Both runs so far gained about 1 coin per 250 rounds over
their last 750 and had not levelled off, so doubling the budget is the cheapest
remaining lever. `frozen_coins` in `eval_curve.csv` is the column to watch;
there is no column called `coins`.

---

# Part 1 — How this works

## The tasks and their gates

**Task 1, `coin-heaven`.** 17x17 board, no crates, no opponents, 50 coins all
visible from step 1, 400 steps. Pure navigation; no bomb is needed, so BOMB is
masked out for this stage.
Gate: 45 of 50 coins with the whole confidence interval above it, and zero
invalid actions. **PASSED** 2026-09-06.

**Task 2, `crate-light` then `classic`.** Crates to blow up, still no opponents.
The agent has to open crates and never blow itself up. The spec calls escaping
bombs crucial and says most tournament losses are self-inflicted.
Gate: suicide rate under 2% over 200 rounds, and 8 of 9 coins on average.
`crate-light` (density 0.35) is a curriculum step we added; `classic` (0.75) is
the tournament setting.
Status: set up and verified, not trained.

**Task 3.** Against `peaceful_agent` and `coin_collector_agent`.
Gate: kills the peaceful agent in 80% of rounds; positive margin vs the coin
collector. Not started.

**Task 4.** Against `rule_based_agent` and self-play.
Gate: mean score above `rule_based_agent` over 200 rounds. Not started.

For task 2 onward, **coins are the wrong thing to watch.** Track
`suicidal_bombs` and `crates`.

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

## The six configurations

Defined in `tools/sweep.py`. All six share the same base, then each changes
**exactly one thing** from `baseline`. That is what makes this an ablation
rather than six unrelated experiments.

Shared by all six: BOMB masked out of the action set, 13x13 egocentric view,
one CPU thread each, `coin-heaven` scenario.

| Config | What differs from baseline | Question it answers |
|---|---|---|
| `baseline` | nothing | The reference the others are measured against |
| `no_shaping` | `DQN_SHAPING=0` | Does the distance-to-coin bonus help at all? |
| `global_view` | `DQN_VIEW=global`, the full 17x17 board | Is the egocentric crop helping or hurting? |
| `lr_2e4` | `DQN_LR` 5e-4 -> 2e-4 | Is it learning too fast to stay stable? |
| `batch32` | batch 32 every 4 steps, not 128 every 16 | Do smaller, more frequent updates learn better? |
| `old_shaping` | the pre-fix reward shaping, deliberately | How much did the shaping fix actually change? |

`batch32` looks like two changes but is one: 32 x (1/4) and 128 x (1/16) both
consume 8 samples per environment step, so the data seen is identical and only
the update rhythm differs. `old_shaping` looks like three
(`DQN_SHAPING_GAMMA=0.95`, `DQN_SHAPING_SCALE=0.1`, `DQN_SHAPING_CAP=999`) but
is one coherent setting: the discounted potential with no distance cap.

`baseline` concretely means: gamma 0.95, learning rate 5e-4, batch 128 every 16
steps, replay buffer 50,000, target network synced every 1000 updates, shaping
on with gamma 1.0 / scale 0.05 / cap 10.

## What one run costs

Every configuration gets an identical budget.

| | |
|---|---|
| Rounds | 3,000 |
| Environment steps | ~1,200,000 |
| Weight updates | 75,000 (300,000 for `batch32`) |
| Transitions drawn from the buffer | ~9,600,000 |
| Each experience reused | ~8 times |
| Exploration | epsilon 1.0 -> 0.05 over the first 1,500 rounds |
| Checkpoints saved | every 250 rounds, so 12 per run |
| Replay buffer memory | ~100 MB per run (uint8) |

Measured wall clock, six in parallel on an M3 Max, one core each:

```
old_shaping   36.9 min      baseline    44.9 min
lr_2e4        44.6 min      no_shaping  45.0 min
global_view   50.5 min      batch32     51.3 min
```

Total elapsed 51 minutes rather than about five hours sequentially.

Two of those times mean something. `batch32` is slowest because 300,000 update
calls carry four times the per-call overhead for the same amount of data. And
`old_shaping` finishing eight minutes early is a result, not noise: a round ends
as soon as every coin is collected, so a shorter run means it was actually
clearing boards. That was visible in the wall clock before it was visible in the
scores.

## What each file is

| File | What it is | Trust it? |
|---|---|---|
| `agent_code/dqn_agent/training_log.csv` | One row per round **during** training | For debugging only. NOT a learning curve. |
| `agent_code/dqn_agent/eval_curve.csv` | Frozen model measured at each checkpoint | **Yes. This is the real curve.** |
| `agent_code/dqn_agent/checkpoints/dqn-rNNNNNN.pt` | Weights snapshot at round NNNNNN | The raw material for the curve |
| `agent_code/dqn_agent/dqn-model.pt` | Working copy the game loads | Overwritten by any training run. Not an archive. |
| `models/*.pt` | **The measured models, tracked in git** | The only copies that survive a clean checkout |
| `agent_code/dqn_agent/logs/dqn_agent.log` | Every action and its Q values | For diagnosing weird behaviour |

## The tools

| Script | What it does |
|---|---|
| `tools/evaluate.py` | Measures one agent frozen, against a gate. Works on any agent, not just the DQN. |
| `tools/train_curve.py` | One continuous training run with checkpoints, then an honest frozen curve from them. |
| `tools/compare.py` | Evaluates every sweep configuration frozen, in parallel, into one table. Applies each config's own settings, so a global-view model is not evaluated with an egocentric network. |
| `tools/confirm.py` | Re-measures the leaders over 200 rounds instead of 40, and only calls PASS when the whole interval clears the bar. |
| `tools/sweep.py` | Runs several configurations at once, one core each. |
| `tools/watch_sweep.py` | Progress of a running sweep, read off disk from another terminal. |
| `tools/profile_step.py` | Where a training round's time actually goes. |

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

```
round, steps, score, coins, invalid_actions, waited,
crates, good_bombs, useless_bombs, suicidal_bombs,
reward_sum, epsilon, mean_loss, buffer
```

Health checks, worth a glance on any run:

- `epsilon` should fall smoothly from 1.0 to 0.05 across the whole run. If it
  jumps back up, something restarted. That was bug #2.
- `buffer` should climb to 50000 and stay. If it drops back to ~400, the replay
  memory got wiped. Same bug.
- `invalid_actions` should be 0 from round 1 with legal-move masking on. Anything
  else means `DQN_LEGAL_MASK=0` or a bug in `legal_actions`.

For task 2, these are the ones that matter:

- `suicidal_bombs` must fall toward 0. It counts bombs dropped where no escape
  route existed, judged before the agent moves, so it is not just "died".
- `crates` must rise. Together they say "learned to bomb usefully and live".
- `good_bombs` versus `useless_bombs` says whether it is bombing on purpose.

If the header ever disagrees with the rows, the file was written by an older
version; `setup_training` now rotates a mismatched log aside rather than
appending wider rows under a narrower header.

## Every switch

Set on the command line; nothing needs a source edit. Official games set no
environment at all, so every default here must be the one we want to ship.

| Variable | Default | Effect |
|---|---|---|
| `DQN_VIEW` | `ego:13` | `global` for the whole board. **Stored inside the checkpoint**, so a saved model rebuilds its own architecture. |
| `DQN_ACTIONS` | all six | Curriculum restriction, e.g. `UP,RIGHT,DOWN,LEFT` for task 1. |
| `DQN_LEGAL_MASK` | `1` | Drop moves the game would reject. `0` for the ablation. |
| `DQN_EXPLORE_BOMB` | `0.10` | Probability a random exploratory action is BOMB. Shapes exploration only. |
| `DQN_GAMMA` | `0.95` | Discount. |
| `DQN_LR` | `5e-4` | Adam learning rate. |
| `DQN_BATCH` / `DQN_TRAIN_EVERY` | `128` / `16` | Batch size and update interval. Their ratio is the replay ratio. |
| `DQN_BUFFER` / `DQN_LEARN_START` | `50000` / `2000` | Replay capacity, and transitions before the first update. |
| `DQN_TARGET_UPDATE` | `1000` | Gradient steps between target syncs. |
| `DQN_EPS_START` / `DQN_EPS_END` / `DQN_EPS_DECAY` | `1.0` / `0.05` / `400` | Set decay to about half the rounds. |
| `DQN_SHAPING` | `1` | `0` for the ablation. |
| `DQN_SHAPING_SCALE` / `_GAMMA` / `_CAP` | `0.05` / `1.0` / `10` | See 2026-09-05: the discounted, uncapped version paid the agent to idle. |
| `DQN_CHECKPOINT_EVERY` | `0` | Numbered snapshots. The curve tools set it. |
| `DQN_MODEL_FILE` | agent's own | Point evaluation at one specific checkpoint. |
| `DQN_RESUME` | `0` | Continue from existing weights. |
| `DQN_PLAY_EPSILON` | `0.0` | Noise at play time. A patch, not a fix. |
| `DQN_THREADS` | `1` | One thread per process; run processes in parallel instead. |
| `DQN_SEED` | `-1` | Set for a reproducible run. |

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

## 2026-09-09 (night, 3) — Training with the shield on makes the agent worse

Retrained 3000 rounds on `crate-light`, identical to the previous run except
that `choice_mask` was masking unescapable bombs throughout. The mask worked
exactly as designed: `suicidal_bombs` is **0 across all 3000 rounds** (it was
0.11 per round before) and invalid actions stayed at 0.

It did not help. Both runs evaluated masked, 300 frozen rounds each:

| checkpoint | trained *unmasked* (run 1) | trained *masked* (run 3) |
|---|---|---|
| r2500 | 4.03 coins, **7.67%** suicide | 3.99 coins, **24.33%** |
| r2750 | 5.88 coins, **9.00%** | 5.96 coins, **21.00%** |
| r3000 | 6.94 coins, **5.67%** | 6.55 coins, **11.33%** |

Coin collection is statistically identical at every checkpoint. The suicide rate
is two to three times worse for the shielded agent, at every checkpoint.

**Why.** The mask only blocks bombs with *no* escape. It does nothing about a
bomb whose escape exists but is a precise four-move sequence. Run 1 learned
caution about those the hard way, by dying, and that caution transfers. Run 3
never saw a single suicidal bomb, so it learned nothing about bomb safety at
all, and leans on a shield that does not cover the case it actually fails on.

This is the shielded-RL failure mode in miniature: shield during training and
you get a policy that depends on the shield rather than one that has learned the
constraint. It is worth a paragraph in the report, because it is a result rather
than an anecdote: two runs, one variable, three checkpoints, 300 evaluation
rounds each, and the coin axis controls for "run 3 is simply behind".

**Change.** `DQN_SUICIDE_MASK` now defaults to off during training and on at
play time, with the explicit override kept for the ablation. The deaths are the
teaching signal; the mask is a safety net over a policy that already learned
most of the lesson.

Run 3's model files are kept but are not the best agent. Note that run 3
overwrote `checkpoints/` and `dqn-model.pt`, so run 1's r3000 weights survive
only as `models/task2-run1-r3000-ego13.pt`.

### Also worth recording

Neither run has converged. Both gained roughly a coin per 250 rounds over their
last 750 rounds and neither levelled off. Every diagnosis so far has been a
reward or masking bug, so it is worth stating plainly that the next step is not
another bug hunt: it is 6000 rounds instead of 3000.


## 2026-09-09 (night, 2) — Masking unescapable bombs, measured before retraining

`choice_mask` now removes `BOMB` when `escape_exists` is false, alongside the
existing legal-move mask. `DQN_SUICIDE_MASK=0` for the ablation. The check is
one BFS bounded by `BOMB_TIMER`; measured decision time went from 1.00 to
1.01 ms against a 500 ms budget.

### It was worth measuring before retraining

Same weights, same 300 frozen rounds on `crate-light`, only the mask changed:

| r3000 | coins | steps | suicide | reach 8+ | invalid | ms/step |
|---|---|---|---|---|---|---|
| mask off (as trained) | 6.49 +/-0.34 | 340.5 | 15.67% | 51.3% | 0.00 | 1.00 |
| **mask on** | **7.05 +/-0.24** | **382.2** | **4.33%** | 53.7% | 0.00 | 1.01 |

A third of the remaining gap to the gate closed without a single training step.
That is worth remembering as a method: the policy was not wrong about where to
bomb, it was wrong about one narrow class of state that could be excluded
outright.

On the 2000-board opening probe the mask does exactly what it should and nothing
more: the fatal opening bomb goes from 9.30% of boards to **0.00%**, while the
38.85% of boards where the agent correctly opens with a bomb are untouched.

### What is left

Residual deaths over 600 rounds: **6.17%**, at steps
`[6 x7, 12, 12, 19, 20, 21, 28, 30, ..., 155, 225]`. Seven are still the first
bomb, but these are cases where an escape *existed* at drop time and the agent
failed to walk it. The rest are spread evenly through the round. So the mask
removed the provably impossible class and what remains is a real learning
problem, which is the right kind of problem to have.

### The caveat, stated plainly for the report

This is a hand-coded rule, not a learned one, and with it on `SUICIDAL_BOMB`
(-3.0) can never fire during training. That is deliberate. The argument is the
same one that justifies the legal-move mask: it encodes a rule the framework
already enforces rather than a strategy, and it never says a bomb is a good
idea, only that this particular bomb is a certain death. The ablation switch
exists so the report can show the difference rather than assert it.

The counter-argument is real and should be acknowledged: an agent that needs the
mask has not learned bomb safety, it has been given it. The 3000-round evidence
is that it does not learn it on its own (42% error rate on the fatal openings
after a full run), because the state is uncommon, the punishment lands four
steps late, and the same action is correct on the other 78% of boards.


## 2026-09-09 (night) — The reward fixes worked; the deaths are the opening move

**Result.** Same command, same 3000 rounds on `crate-light`, only the reward
function changed. Frozen evaluation went from 0.0 coins to 6.62 at the last
checkpoint, and every intermediate number moved with it.

| | before the fixes | after |
|---|---|---|
| frozen coins @ 3000 | 0.00 | **6.62** |
| crates / round (last 300, training) | 1.33 | **29.65** |
| good bombs / round | 0.62 | **13.11** |
| mean reward / round | not logged | +31.6 |
| invalid actions, all 3000 rounds | 0 | **0** |

Training-time progression of the new run:

```
rounds        eps   steps   coins  crates   good  useless  suicid   reward    loss
1-300       0.905    18.5    0.01    1.85   0.91     0.21   0.103    -9.55  0.2684
1201-1500   0.145    47.4    0.18    4.65   2.40     2.76   0.217    -5.07  0.3040
1501-1800   0.050   140.0    0.70    9.90   5.36     7.09   0.213    +4.91  0.2446
2101-2400   0.050   144.0    1.77   23.20  11.77     4.03   0.083   +23.75  0.0962
2701-3000   0.050   137.8    3.17   29.65  13.11     3.66   0.110   +31.64  0.0687
```

The collapse at round 1500 that killed the previous run did not happen. Crates
rise straight through the point where epsilon anneals and keep rising. Useless
bombs peak at 7.09 and then fall to 3.66, which is the agent learning bomb
placement rather than learning to stop bombing.

### Frozen evaluation, 200 rounds per checkpoint

```
r2500: coins 3.91 +/-0.35 | steps 345 | suicide 14.0% | invalid 0.00
r2750: coins 5.24 +/-0.46 | steps 331 | suicide 17.5% | invalid 0.00
r3000: coins 6.25 +/-0.41 | steps 344 | suicide 14.0% | invalid 0.00
```

Coins are climbing steeply and had not levelled off when the run ended. The
suicide rate is flat. **These are two independent problems**, and only the coin
one is solved by training longer.

### Where the coins actually are, 300 rounds at r3000

```
died       n= 45  coins 0.47  steps  21.6  reach 8+:  0.0%
survived   n=255  coins 7.47  steps 398.2  reach 8+: 59.2%
ALL        n=300  coins 6.42  steps 341.7  reach 8+: 50.3%
```

Coin distribution over 300 rounds: `{0: 38, 1: 2, 2: 5, 3: 8, 4: 5, 5: 9,
6: 29, 7: 53, 8: 80, 9: 71}`. It is bimodal. Either the agent dies immediately
with nothing, or it plays a good round. There is almost nothing in between.

**Half of all rounds already meet the coin half of the gate.** Removing the
deaths alone moves the mean to 7.47 against a target of 8.

### Root cause: it opens with a bomb it cannot escape

Death step quartiles: `[5, 5, 5, 5, 5, 5, 7, 34, 68]`. `BOMB_TIMER` is 4, so a
bomb dropped on step 1 explodes at the end of step 5. Almost every death is the
agent's own first bomb.

Probed directly: build 2000 `crate-light` boards, put the network in the start
state, take its greedy action, and separately ask `escape_exists` whether a bomb
there is survivable.

```
boards tested                          2000
opening bomb would be unescapable      22.0%
agent picks BOMB as its first action   48.1%
  ... and it is escapable              38.9%
  ... and it is NOT escapable           9.3%   <-- guaranteed death
```

The start corner is cleared of crates by `build_arena`, but the corridor out of
it is not, and with 35% density there is often no tile at distance 4 reachable
in the 4 steps the timer allows. The agent has learned this *partially*: of the
22% of boards where the opening bomb is fatal it correctly declines on 58% of
them. On the other 42% it dies.

9.3% of boards is a guaranteed death with a frozen policy and no exploration.
The measured suicide rate is 15%. So the opening move alone is about two thirds
of every death the agent suffers.

### The fix, and the honest caveat

Mask `BOMB` in `choice_mask` whenever `escape_exists` is false, exactly as
illegal moves are already masked. `escape_exists` is already written, already
unit-tested, and one BFS per step is nothing against a 0.99 ms decision time and
a 500 ms budget.

The caveat for the report: this is a hand-coded rule, not a learned one.
`SUICIDAL_BOMB` at -3.0 was supposed to teach it and did not, because the state
is uncommon, the punishment arrives four steps late, and the same action is
correct on 78% of boards. It belongs in the same category as the legal-move
mask: it encodes the rules of the game, not a strategy. Worth saying out loud
rather than hiding.

### Housekeeping

`_open_log` only rotates the log aside when the header changes, so this run
appended to the previous one and `training_log.csv` now holds two runs with the
round counter restarting at line 3002. Archived correctly as
`experiments/runs/2026-09-09-task2-crate-light-3000r-fixed.csv`; the rotation
rule should probably also trigger when round 1 reappears.


## 2026-09-09 (evening) — Four reward fixes, with the arithmetic

The agent was behaving correctly under a reward function that asked for
nothing. Expected value of a five-step window, gamma 0.95, assuming a bomb
breaks two crates and 5% of bombs kill it:

| | bomb 2 crates | do nothing | margin |
|---|---|---|---|
| before | -0.75 | -0.32 | **-0.44** |
| after | +2.35 | -0.32 | **+2.67** |

Still positive if 30% of bombs kill it (+1.42). That is the check that was
missing before the first 3000-round run: the reward design was never priced
against the alternative of standing still.

### Fix 1: the potential needs a target that exists

`_distance_to_target` returns the distance to the nearest collectable coin, or
if there is none, to the nearest tile from which a bomb would break a crate.
The coin-only version returned None on every task-2 board.

### Fix 2: your own bomb is not "walking into danger"

`ENTERED_DANGER` (-0.6) is suppressed on the step a bomb is dropped. Dropping
one necessarily puts the agent in its own blast; charging for that penalises
bombing twice, once through the bomb verdict and again for its unavoidable
consequence.

### Fix 3: escaping is progress, not stalling

Found while writing out the arithmetic rather than by running anything.
`CLOSER_TO_SAFETY` could **never fire**. It compared steps-until-lethal before
and after, but a blast shares one countdown, so that number falls by exactly one
per step wherever the agent moves inside it, and `now > was` is impossible.
Every step of a correct four-step escape therefore fired `STAYED_IN_DANGER` at
-0.4, taxing a successful escape -1.2.

Progress is now measured with `steps_to_safety`, a breadth-first distance to the
nearest tile still safe on arrival. Verified along a corridor: standing at
distance 1, 2, 3, 4 from the blast edge gives 4, 3, 2, 1 moves to safety, and
each step of a correct escape now registers as `CLOSER_TO_SAFETY`.

### Fix 4: rebalance

`GOOD_BOMB` 0.4 -> 1.0, `CRATE_DESTROYED` 0.3 -> 0.5, `COIN_FOUND` 0.2 -> 0.5,
`KILLED_SELF`/`GOT_KILLED` -6.0 -> -5.0. Suicidal bombs were already near zero,
so the discipline is coming from `SUICIDAL_BOMB` (-3.0); -6.0 on top of it was
what made bombing look like a bad trade.

### Method note

Three of these four came from writing the reward out as arithmetic, not from a
training run. The 3000-round run that produced 0.0 cost about 45 minutes and
told us less than ten minutes of algebra. Worth pricing a reward change against
the do-nothing baseline before spending a run on it.


## 2026-09-09 — Task 2 trains to a score of zero, and why

**Result.** 3000 rounds on `crate-light`. Frozen evaluation: **0.0 coins at all
twelve checkpoints**, 400 steps, 0 invalid actions.

Training-time progression:

```
rounds        eps     steps   crates  good_bombs  suicidal  invalid
1-300       0.810     46.6      1.96        0.96      0.08     0.00
601-900     0.430    125.1      3.14        1.50      0.06     0.00
901-1200    0.240    203.4      3.23        1.60      0.04     0.00   <- peak
1201-1500   0.050    332.2      2.31        1.12      0.08     0.00
1501-1800   0.050    379.6      1.20        0.59      0.03     0.00
2701-3000   0.050    379.4      1.33        0.62      0.05     0.00
```

**What worked.** Survival. Steps went 47 -> 380 and suicidal bombs stayed near
zero throughout, so the escape detection and the -3.0 penalty did their job.
Invalid actions were 0.00 from the first round, so legal-move masking held.

**What failed.** Crates peaked at 3.2 per round while exploration was still
forcing bombs, then collapsed to 1.2 the moment the policy went greedy. The
agent actively chose to bomb less. It learned to survive by doing nothing.

### Root cause: the shaping is inert on a crate board

`_potential` is `-scale * BFS distance to the nearest coin`, and
`game_state["coins"]` lists only *collectable* coins. Measured across the three
scenarios at step 1:

| scenario | coins | crates | visible at step 1 |
|---|---|---|---|
| `coin-heaven` | 50 | 0 | **50** |
| `crate-light` | 9 | 56 | **0** |
| `classic` | 9 | 119 | **0** |

Every coin starts inside a crate. So on any task-2 board the potential is 0 at
every step and the shaping term contributes exactly nothing. The whole dense
reward signal that carried task 1 is switched off, and what remains is a -0.02
step penalty, a -6.0 death penalty, and bomb events whose expected value is
marginal. Standing still is a rational optimum under that reward function.

This is the same class of error as 2026-09-05: the reward function was measured
against the wrong scenario. It was correct for task 1 and silently degenerate
for task 2.

### Fixes to apply, none applied yet

1. **Make the potential target crates when no coin is visible.** Phi should be
   the distance to the nearest coin if one is collectable, else the distance to
   the nearest tile from which a bomb would break a crate. This restores dense
   guidance toward the thing actually worth doing.
2. **Stop double-penalising your own bomb.** Dropping one necessarily puts the
   agent in its own blast, which fires `ENTERED_DANGER` (-0.6) on top of the
   bomb verdict. Suppress that event for the agent's own freshly dropped bomb.
3. **Rebalance.** `GOOD_BOMB` +0.4 and `CRATE_DESTROYED` +0.3 against
   `KILLED_SELF` -6.0 makes bombing marginal, and the crate reward arrives four
   steps later discounted by gamma while the death is immediate.

Do not train again until at least fix 1 is in. The outcome will not change.

### Arithmetic worth keeping for the report

At 0.35 crate density there are ~56 crates and 9 coins, so roughly 16% of crates
hide a coin. Reaching the gate's 8 of 9 coins means destroying most of the
crates on the board. At the observed peak of 3.2 crates per round the agent
would find about 0.5 coins. The gate is not close, and the target behaviour is
"bomb constantly and survive", not "bomb occasionally".


## 2026-09-08 — Task 2 groundwork, and two things measured rather than assumed

### Bomb timing, measured against a live game

The danger channel had never run with a real bomb on the board. Rather than
re-read the code, a probe agent dropped one and recorded what it saw:

```
step 1  drop            state shows no bomb yet
step 2  timer 3         explosion_map 0
step 3  timer 2         explosion_map 0
step 4  timer 1         explosion_map 0
step 5  timer 0         detonates at the end of this step, KILLED_SELF
step 6  no bomb         explosion_map 1, still lethal at end of step
step 7  no bomb         explosion_map 0, genuinely safe
```

So: **a bomb showing timer T is lethal at the end of the step T steps from now**,
and `explosion_map > 0` means lethal at the end of THIS step. Seeing timer T
leaves T+1 actions to get clear. An explosion is visible and lethal for exactly
one step, not two.

The old danger map was directionally right. It is now derived from an explicit
`steps_until_lethal` grid with those semantics written down.

### Escape detection

`escape_exists(state, extra_bomb=...)` walks outward and asks whether a tile
that is still safe on arrival can be reached in time. Unit-tested on hand-built
boards: dead ends of 3 and 4 tiles fail, a straight corridor passes (a tile 4
away is already outside a power-3 blast and reachable in 4 moves), a corner at
distance 2 passes, and a corridor whose exit is blocked by a second bomb fails.

One test of mine was wrong before the code was: I expected a corner 5 tiles away
to be unreachable, forgetting `BOMB_POWER` is 3, so tile 4 is already safe. The
code was right.

### Bomb events and rewards

Following the spec's advice, and Daniela's, that dense rewards beat features and
cost far less: `GOOD_BOMB` (+0.4), `USELESS_BOMB` (-0.3), `SUICIDAL_BOMB` (-3.0),
`ESCAPED_DANGER` (+0.5), `ENTERED_DANGER` (-0.6), `STAYED_IN_DANGER` (-0.4),
`CLOSER_TO_SAFETY` (+0.15). Paired so opposites cancel rather than leaving a
farmable positive, which is the trap that bit us on 2026-09-05.

Verified the classifier discriminates rather than labelling everything:

| scenario | good | useless | suicidal | crates/round |
|---|---|---|---|---|
| `empty` (open board) | 0 | 2 | 0 | 0.0 |
| `loot-crate` (75% crates) | 1 | 0 | 5 | 3.2 |

Open board: every bomb escapable and pointless. Cramped board: most bombs are
genuine suicide. `training_log.csv` now has `crates`, `good_bombs`,
`useless_bombs` and `suicidal_bombs` columns.

### Legal-move masking (from block B)

Illegal moves are removed from the argmax and from exploration. Effect measured
from scratch at epsilon 1.0: **0 invalid actions from the first round** (against
~145 for the old random walk) and 23-28 coins while acting entirely at random
(against ~18). Better random exploration means better data in the buffer.

Because legality is per-state, each transition now stores the successor's legal
mask; without it, untrained noise on illegal actions wins the Double DQN argmax
and poisons the targets, the same failure as the curriculum mask earlier.

Ablation switch: `DQN_LEGAL_MASK=0`.

### Two changes that made task 2 trainable at all

Uniform exploration picks BOMB one time in six, which on a crate-dense board
kills the agent in 3-5 steps, before any experience accumulates. This is bug 0
in a new costume.

1. `DQN_EXPLORE_BOMB` (default 0.10) weights BOMB down during exploration only.
   The framework's own template agent does the same. The greedy policy is
   untouched.
2. A `crate-light` scenario (density 0.35 against the tournament's 0.75) as a
   curriculum step.

Measured effect, 8 rounds from scratch:

| scenario | mean steps | crates/round | suicidal bombs/round |
|---|---|---|---|
| `crate-light` | 42 | 1.6 | 0.0 |
| `loot-crate` | 26 | 3.5 | 0.4 |

Against 3-5 steps before. Note `settings.py` changes do not ship with the agent,
so the tournament is unaffected; the spec explicitly permits custom scenarios.

### The view now travels inside the checkpoint

`DQN_VIEW` was read from an environment variable the tournament never sets, so a
global-view model would have been given an egocentric architecture and failed to
load. Checkpoints now store `{"view": ..., "state_dict": ...}` and `setup` reads
the view before building the network. Verified by saving a global-view model,
clearing the environment, and loading it as the tournament would. Old bare
checkpoints still load via a fallback.

### Housekeeping

`dqn_batch32`, `dqn_lr_2e4`, `dqn_no_shaping` and `dqn_old_shaping` moved to
`_to_delete/`. Their numbers survive in `experiments/sweep_results.csv`, so
nothing is lost by removing them.

`.gitignore` now ignores `agent_code/dqn_*/` but un-ignores `dqn_agent`,
`dqn_baseline` and `dqn_global_view`. Their source and the two winning model
files are tracked (about 4 MB); the 24 remaining checkpoint files are not.

`.vscode/launch.json` has five run configurations, including a "Submission
check" that runs with an empty environment exactly as the tournament will.

### Still not done

Nothing is trained on task 2 yet. The `danger` channel is now correct but the
network has never seen it nonzero during training.


## 2026-09-06 (confirmed) — Gate 1 passed

Re-measured the top checkpoint of each configuration over 200 fresh rounds
(the sweep used 40, giving +/- 2 coins, too wide to claim a threshold, and
taking the max of 72 measurements biases upward).

`baseline` @ 250 rounds passes: 46.06 coins, CI [45.5, 46.7], 0 invalid.

Three configurations are marginal: their means clear 45 but either the
confidence interval dips below it or they commit a few invalid actions.
Reporting those as passes would be overclaiming.

The interesting entry is `global_view` @ 2750: 47.39 coins [46.5, 48.3] in 234
steps, the best score and by far the fastest board clear, held back from a PASS
only by 3.19 invalid actions per round (about 1.4% of its moves). Given it is
also the only configuration that does not degrade with training, it is the
better agent despite the verdict, and it is the one to carry into task 2.

Worth noting for the report: the strict gate I wrote (zero invalid actions) is
my own criterion, not the project spec's. It ranked the weaker agent first. That
is a fair illustration of why the metric definition deserves as much scrutiny as
the result.


## 2026-09-06 — It learns. And two of my conclusions were wrong.

**Result.** Re-ran the six-config sweep with the cache fix. Best frozen scores
per config: 40.9 to 47.9 coins, against ~5 before. `mean_steps` drops below 400
for the first time, i.e. the board actually gets cleared.

Six checkpoints provisionally meet gate 1 (>= 45 coins, 0 invalid actions) on 40
evaluation rounds:

| config | rounds | coins | steps |
|---|---|---|---|
| baseline | 250 | 47.35 +/- 1.27 | 333 |
| lr_2e4 | 250 | 47.00 +/- 0.72 | 367 |
| global_view | 2250 | 46.35 +/- 2.12 | 280 |
| old_shaping | 250 | 45.52 +/- 2.16 | 342 |
| lr_2e4 | 500 | 45.48 +/- 1.76 | 335 |
| old_shaping | 500 | 45.12 +/- 1.99 | 373 |

Not yet a claim: 40 rounds gives +/- 2 coins, and taking the maximum of 72 noisy
measurements biases upward. `tools/confirm.py` re-measures over 200 rounds.

### Reversal 1: the global view is better, not worse

`global_view` is the ONLY configuration that improves monotonically across the
whole run, and the only one whose `mean_steps` falls steadily:

```
250:31.7(394 steps)  1000:37.9(362)  1750:42.9(301)  2250:46.4(280)  2750:47.9(236)
```

Every egocentric config peaks at 250-500 rounds and then decays. Likely
explanation: the 13x13 crop discards coins outside the window. Early in a round
coins are dense and the crop is enough; late in a round, when few coins remain
and they are far away, the egocentric agent is simply blind and cannot improve.
The global view always sees every coin.

I previously read `global_view`'s high invalid-action count during training as
evidence the crop was helping. That was wrong: it was behind on epsilon at the
time. The corrected reading is the opposite.

### Reversal 2: the "broken" shaping outperforms my fix

Final frozen scores: `old_shaping` 44.5, `baseline` (fixed shaping) 32.2.
`old_shaping` is also far more stable late in training.

The arithmetic in the 2026-09-05 entry was correct: the discounted potential
really does pay +0.065 per two steps for hovering. But it was diagnosing the
wrong failure. With usable transitions, the agent can find something better than
hovering, and the distance-proportional term evidently provides a stronger
long-range gradient toward coins than the capped, undiscounted version.

Being right about the arithmetic and wrong about the consequence is worth
writing up as-is.

### The real open problem: instability

Most configurations peak early and then decay badly, with invalid actions
climbing from 0 to 40-80 per round:

```
baseline   250:47.4(0 invalid) -> 3000:32.2(46 invalid)
batch32    250:41.0(0)         -> 3000:27.6(54)
lr_2e4     250:47.0(0)         -> 3000:36.3(24)
```

Peak performance at 250 rounds means the best policy came from a period when
epsilon was still ~0.84. Continued training destroys it. That is textbook DQN
divergence and it is now the main thing to fix. Candidates: lower learning rate,
more frequent target syncs, gradient-clipping already on, or simply keeping the
best checkpoint rather than the last one.

Only `global_view` is immune, which is further evidence the egocentric crop is
the source of the problem rather than a fix for it.


## 2026-09-05 (evening) — Bug #4: every next_state was its own state

**This is the one that mattered.** All previous results are void.

**Symptom.** Six configurations, 3000 rounds each, all landed between 2.95 and
5.20 frozen coins with overlapping confidence intervals. `mean_steps` 400.0
everywhere. Nothing differed: not shaping on/off, not the fixed shaping vs the
broken one, not egocentric vs global view, not learning rate, not batch size.

Six variables changed and *nothing* moved. That is not six weak effects, that is
a common cause upstream of all of them.

**Cause.** The `_encode` cache added on 2026-09-03 as a speed optimisation, keyed
on `(round, step)`. But `environment.do_step` increments `self.step` exactly once,
at the top, before `poll_and_run_agents`. Both the state passed to `act()` and
the successor state passed to `game_events_occurred()` are therefore built with
the SAME step number. Every successor lookup hit the predecessor's cache entry.

**Evidence.** Instrumented a real 2-round training run: 800 transitions
examined, **800 shared a cache key, 800 had next_state == state. 100%.**
After removing the cache: 35.2% identical, which is exactly right, since at
epsilon 1.0 the agent bumps a wall about a third of the time and a failed move
genuinely leaves the board unchanged (it matches the ~145 invalid actions per
400 steps measured on day one).

**Why it produced exactly these symptoms.** With `s' = s` the target becomes
`r + gamma * max_a' Q(s,a')`: self-referential, no information flows backwards
from future states. Only immediate rewards are learnable. Wall avoidance is an
immediate -0.5, so it was learned to near zero. Coin collection needs value to
propagate across steps, so it was unreachable. This also retroactively explains
2026-09-03: the "frozen policy loops" observation was real, but the loop was a
symptom of a value function that could not represent anything multi-step.

**Fix.** Cache removed entirely, not repaired. Profiling put this encoding at
~2% of a training round, so it was never worth the risk. There is a comment in
`_encode` saying not to reintroduce it.

**Lesson for the report.** Three earlier "fixes" (egocentric view, truncation
handling, reward shaping) were all aimed at symptoms of this. Two of them were
defensible improvements on their own terms and the shaping analysis was
arithmetically correct, but none of them could have worked while the replay
buffer contained no usable successor states. The diagnostic that finally worked
was noticing that six independent variables all produced the identical result,
which points upstream rather than at any of them.

**Still unknown.** Whether the DQN learns once the data is correct. A validation
run is in progress.


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
