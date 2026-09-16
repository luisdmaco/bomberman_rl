# DQN arm: working log

Owner: Luis. Everything about the deep Q-network agent lives here.
Read Part 1 once to know what you are looking at. Part 2 is the running history,
newest first.

---

## STATUS (2026-09-15, targeted hunting fine-tune prepared; final agent choice open)

**C r1000 remains the measured, submission-tested fallback in `models/`.
`agent_code/dqn_agent/dqn-model.pt` currently contains score-profile r1500, not
C.** Score-profile r1250 passes the required mean-score gate at N=1000 and is
much safer, but it is not a dominant per-round winner; do not install it until
the remaining gates and final trade-off are decided.
Both finalists pass gate 4 over 1000 frozen rounds. C is effectively tied with
A on tournament score and is substantially better on solo coins and both
task-3 measures. A is safer, but that safety does not buy a higher score.

That selection is an average-score result, not a claim that C usually wins an
individual animated round. The user watched C and A on the same seed and both
looked weak. That is consistent with the frozen matrix: C dies by suicide in
32.5% of crowded rounds, A in 17.4%, and each gets a kill in only about 27%.
Both are noisy policies, not robust round-by-round winners.

Final-candidate measurements. Solo and tournament shape (us + 3 `rule_based`)
are 1000 rounds. Task 3 is 300 rounds against `peaceful` and 400 against
`coin_collector`. All runs are frozen.

| model | solo coins | DQN tournament score | weakest task-4 margin | suicide (crowded) | invalid (crowded) |
|---|---|---|---|---|---|
| **C r1000, selected** | **7.70 +/- 0.11** | 3.79 +/- 0.19 | +0.54 +/- 0.28 PASS | 32.50% (95% upper 35.47%) | 2.25 +/- 0.12 |
| A r3750 | 6.33 +/- 0.13 | **3.81 +/- 0.19** | **+0.63 +/- 0.27 PASS** | **17.40% (95% upper 19.87%)** | **0.75 +/- 0.06** |

| model | kills `peaceful` in a round | margin vs `coin_collector` |
|---|---|---|
| **C r1000, selected** | **76.7% [95% 71.6%, 81.1%] FAIL** | **+0.13 +/- 0.40 FAIL** |
| A r3750 | 59.3% [95% 53.9%, 64.7%] FAIL | -0.82 +/- 0.38 FAIL |

C still bombs much more solo: 45.66 +/- 0.76 per round against A's
31.16 +/- 0.49. That is the mechanism behind both its higher solo score and
its worse crowded-room safety.

### Gate scoreboard

| task | gate | status |
|---|---|---|
| 1, `coin-heaven` | 45 of 50 coins, 0 invalid | **PASSED** 06.09, 46.06 coins |
| 2, `classic` survival | suicide < 2% | **PASSED**, C 1.00% over 1000 rounds, 95% upper 1.83% |
| 2, `classic` coins | 8 of 9 | not passed, C **7.70 +/- 0.11** over 1000 rounds |
| 3, vs `peaceful` | kill in >= 80% of rounds | not passed, C **76.7% [95% 71.6%, 81.1%]** |
| 3, vs `coin_collector` | positive margin | not passed, C +0.13 +/- 0.40; historical best +0.17 +/- 0.47 |
| **4, vs `rule_based`** | **mean score strictly above** | **PASSED**, both finalists at 1000 rounds; every paired CI above zero |

Tasks 1 to 3 and their thresholds are our own curriculum. Task 4 is the spec's.

### START HERE

C remains the measured, submission-tested fallback in `models/`. Do not repeat the previous
training recipe: its reward function pays far more for demolition than the
tournament scoreboard does. An optional `DQN_REWARD_PROFILE=score` final-stage
profile is now implemented and verified, while the historical `curriculum`
profile remains the default. The 1500-round fine-tune and r1250 confirmation
are complete. At N=1000 r1250 passes the required average-score gate with
margins +0.53 to +0.56 and 7.8% suicide, but finishes strictly first in only
25.9% of rounds. The next DQN experiment is prepared as the isolated
`score_hunt` profile: +0.5 for an escapable bomb whose blast currently contains
an opponent, while crate-only bombs remain neutral. Resume from archived r1250
with augmentation off, then select from the frozen curve using score,
strict-first rate, kills and suicide. Do not continue the old `score` recipe.
The latest remote Q agent (`c0f36d7`) is also available as an exact unstaged
snapshot, and a shared frozen comparison is ready. The team should confirm that
`model.npy` is the intended Q checkpoint, then run
`tools/compare_team_agents.py --rounds 200 --seed 211`. Project-wide work also remains: choose the best
overall tournament entry, make the repository public, commit and push the final
artifacts, build a clean submission zip, test that exact zip in Docker, and
write the report. See the 2026-09-13 entry below.

For a visual smoke test, VS Code now has `Play selected DQN (GUI, 3 random)` and
`Play selected DQN (GUI, 3 rule based)`. Both are one frozen round, use the
project virtual environment, and explicitly clear every play-time `DQN_*`
override so they load the installed checkpoint with tournament defaults.
The 14.09 rule-based GUI run loaded the correct model and had no timing errors,
but was stopped before the round wrapped up. It is a smoke test, not evidence
for or against performance; the 1000-round frozen matrix remains authoritative.
Two additional VS Code launchers compare C and A on the same seed-77 classic
board against three rule-based opponents. They point at the archived files
directly, so neither launcher changes the installed `dqn-model.pt`.

### The two candidate models, archived

| file | what it is |
|---|---|
| `models/task4-C-solo-refresh-r1000-ego13.pt` | **selected**. Solo 7.70 +/- 0.11; gate 4 passes; task-3 leader; 32.50% crowded suicide (95% upper 35.47%) |
| `models/task4-A-mixed-r3750-ego13.pt` | safer finalist. Solo 6.33 +/- 0.13; gate 4 passes; 17.40% crowded suicide (95% upper 19.87%) |
| `models/task4-classic-r3250-ego13.pt` | the specialist both were built from. Gate 4 +0.78 at 1000 rounds, solo coins only 4.95 |
| `models/task2-classic-r3750-ego13.pt` | the task-2 all-rounder. Solo coins 7.65, fails gate 4 at -1.02 |

All four are gitignored and need `git add -f`.

### The three lessons this project ends on

**Signal density decides what can be learned, and it is a property of the
opponent.** Same reward table, same 4000 rounds: `rule_based_agent` walks into
your blasts and gave 258 kills rising from 0.010 to 0.147 per round;
`coin_collector_agent` ignores you and gave 81, flat. The first taught the agent
to fight, the second taught it to hide.

**Training against one opponent buys a specialist, and the mechanism is
measurable.** The specialist's whole solo deficit was bombing rate: 26.5 bombs
against 47.6. Every non-training measure regressed, invisibly from the training
curve. Measure every gate after every run.

**A tiny corrective run can undo it.** 1500 solo rounds at a fifth of the
learning rate, 7.6 minutes, recovered solo coins from 4.95 to 7.87 while keeping
the gate. I predicted it would swing back and lose the gate. It did neither.

### What is left, in order

1. Verify the teammates' current feature-based Q-learning model and make the
   team-wide tournament choice. C is only the selected DQN checkpoint.
2. Commit and push this branch, force-adding the reported ignored model files,
   and make the repository publicly accessible.
3. Build a minimal `final-project-agent-code.zip` from the winning agent folder,
   test that exact payload in Docker, and upload it for the 17.09 pre-run.
4. Final agent deadline: 21.09 at 21:00. Report deadline: 28.09 at 21:00.

### Reference numbers

| | |
|---|---|
| Task 1 best (frozen) | 47.39 coins in 234 steps |
| `rule_based_agent`, solo `classic` | 8.60 coins, 0.00% suicide |
| `rule_based_agent`, tournament shape | 3.30 mean score, 6.9 invalid actions/round |
| Random walk, `coin-heaven` | ~18 coins |
| Submission dry run | **PASSED 12.09 with selected C**, stock settings, 0 crashes |
| Augmentation ablation | implemented, tested, never run |

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

**Task 3.** Against `peaceful_agent` and `coin_collector_agent`, one opponent
at a time.
Gate: kills the peaceful agent in >= 80% of rounds; positive margin vs the coin
collector.
Status: baseline measured 2026-09-10 with the task-2 weights, neither half
passed. 74.5% and -0.54. Note that the framework's round statistics sum `coins`,
`kills` and `suicides` over *all* agents, so from this task onward
`tools/evaluate.py` cannot answer either gate; use `tools/evaluate3.py`.

**Task 4.** Against `rule_based_agent` and self-play. **This is the only gate the
spec actually requires**; tasks 1 to 3 and their thresholds are our own
curriculum. The spec calls beating `rule_based_agent` the entry ticket.
Gate: mean score above `rule_based_agent` over 200 rounds.
Status: baseline measured 2026-09-10 with the task-2 weights. Not passed:
-1.85 +/- 0.50 one against one, -1.02 +/- 0.40 in the tournament shape.
Measure it with `tools/evaluate3.py`; note the framework renames duplicate
opponents to `rule_based_agent_0/_1/_2`.

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
| `tools/evaluate3.py` | Per-agent, per-round measurement. The only tool that can answer a gate with an opponent on the board, because the framework's own round statistics are summed over all agents. `--gate 3`. |
| `tools/test_others.py` | 41 checks that the agent actually perceives opponents: channel 3, the ego crop, opponents' bombs as danger, opponents' bodies as blocked tiles, and a live game against `peaceful_agent`. |
| `tools/train_curve.py` | One continuous training run with checkpoints, then an honest frozen curve from them. `--opponents` sets who is in the training game; `--eval-opponents` sets who the frozen curve is measured against, which should be the gate's opponents and not the training ones. |
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

With `--opponents`, four more columns appear and `frozen_coins` stops being the
headline, because somebody else is collecting them too:

| Column | Meaning | What good looks like |
|---|---|---|
| `suicide_rate` | Share of rounds ending in our own bomb | Must fall. Baseline vs `coin_collector` is 13.25% |
| `kill_in_round` | Share of rounds with at least one kill | Gate 3 wants >= 80% against `peaceful_agent` |
| `score_margin` | Our score minus the best opponent's, per round | Gate 3 wants this positive |
| `margin_ci95` | Half-width of its 95% interval | At 40 eval rounds this is about +/- 2. The curve is a shortlist, not a measurement. |

A 40-round curve point is noisy enough that the same weights measured twice gave
3.48 coins / 24% suicide and 4.28 coins / 8% suicide. Shortlist from the curve,
then confirm with `tools/evaluate3.py` at 200 rounds or more.

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
| `DQN_MODEL_FILE` | agent's own | Point evaluation at one specific checkpoint. A relative path is resolved **against the repo root, not the shell's working directory**, and a path that does not exist raises. Both are scars; see 2026-09-10. |
| `DQN_RESUME` | `0` | Continue from existing weights. |
| `DQN_OPPONENT_MASK` | `1` at play, `0` in training | Widen the survivability mask: an escape route an opponent can occupy first is not an escape. Measured 2026-09-10: self-kills 20.25% -> 13.25% against `coin_collector_agent`. |
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

## 16.09.2026 - Task-4 hyperparameter sweep: no factor beat the seed

Six configurations, 5000 rounds each from scratch on `classic` against three
`rule_based_agent`s, score reward profile, one core each. Five ran in
parallel in 287 minutes wall clock; the sixth crashed after a minute and was
re-run alone in a further 184 minutes (see below). Every 250-round checkpoint
was then evaluated frozen, training off, 60 rounds per checkpoint, and the
configurations ranked on the mean of the last five checkpoints (300 pooled
rounds each) rather than on the best single checkpoint.

| config | pooled margin | best checkpoint | score | suicide | kill/round |
|---|---:|---:|---:|---:|---:|
| ctrl_s2 | -1.401 | -1.117 @ r5000 | 1.76 | 31.0% | 12.0% |
| augment | -2.008 | -1.611 @ r4750 | 1.36 | 46.7% | 12.0% |
| target250 | -2.190 | -1.772 @ r4250 | 1.32 | 46.7% | 12.0% |
| gamma99 | -2.354 | -1.711 @ r3250 | 1.14 | 53.0% | 9.7% |
| ctrl_s1 | -2.431 | -2.067 @ r5000 | 1.02 | 54.0% | 10.7% |
| global | -4.117 | -3.433 @ r1750 | 0.15 | 6.0% | 1.0% |

**The two controls differ by 1.030 margin.** They share every hyperparameter and
differ only in `DQN_SEED`. That number is the resolution of this experiment, and
almost nothing survives it:

- `DQN_GAMMA=0.99` (-2.354), `DQN_TARGET_UPDATE=250` (-2.190) and
  `DQN_AUGMENT=1` (-2.008) all land *inside* the control band
  [-2.431, -1.401]. None is separated from the control. The README's
  long-standing "raise gamma for tasks 3 and 4" is not supported by this
  measurement, and the 8-fold symmetry augmentation - the cheapest expected win
  going in, and the one with a mechanistic reason to help an undertrained run -
  bought nothing measurable either.
- `DQN_VIEW=global` (-4.117) is the only configuration outside the band, and it
  is 1.686 below the *worse* control. The coin-heaven result that motivated it,
  where global overtook ego:13 after ~1000 rounds and reached 47.9 coins at
  r2750, did not transfer to task 4 at all. The low 6.0% suicide rate does not show that the agent learned bomb safety, with 0.15 score and a kill in 1% of rounds it learned to stand still.

The margin trajectories show that the ego runs were still improving at the
end: ctrl_s2: -3.17, -3.02, -2.22, -1.12 at r1250/2500/3750/5000

The global run stayed almost flat at around -4 throughout. This means that the from-scratch agents were probably still undertrained
rather than fully converged. However, after 5000 rounds they were still
1.95 margin points behind the existing candidate. In earlier experiments,
the curves usually flattened before closing a gap of this size.

**Nothing here is a submission candidate** - The current best agent is still the score-profile r1250 agent. It has a
+0.55 margin, a score of 3.437, and a 7.8% suicide rate over N=1000.

The best configuration from this sweep reaches only -1.40 margin and has
a 31% suicide rate. The 5000 from-scratch rounds therefore did not reproduce the performance that the curriculum had already achieved (task 1 coins, then task 2 crates, then a score-aligned fine-tune), the suicide rates alone say the from-scratch agents never learned bomb safety.

This settles the 14.09. selection. Two runs that differ only in `DQN_SEED`
end 1.03 margin apart after 5000 training rounds, measured over 300 pooled
evaluation rounds each. A checkpoint picked because its 40-round reading was
+1.42 is therefore picked from inside the noise.

The later N=1000 evaluation came back at +0.55, and that correction was
predictably downward rather than merely different. Taking the best of six
noisy checkpoints is an upward-biased estimator: the winner is the one whose
noise happened to point up, so its value is expected to shrink on any
re-measurement. A larger confirming sample alone would have been as likely to
correct upward.

### The augmentation arm crashed on Windows first

The `augment` arm crashed about one minute after starting, at the first gradient step:

```
model.py:224  actions[idx] = _PERM[t][actions[idx]]

RuntimeError: Index put requires the source and destination dtypes match,
got Long for the destination and Int for the source
```

The problem was caused by a difference in data types. `np.arange(N_ACTIONS)`
uses `int32` on Windows and `int64` on Linux. The `actions` tensor from
`ReplayBuffer.sample` is always `torch.int64`. Because of this, the 8-fold dihedral augmentation works on Linux but crashes
on Windows. `_INV` did not have this problem because `np.argsort` returns
`int64` on both platforms.

The issue was fixed by setting the data type explicitly in
`action_permutation`. After that, `tools/test_augment.py` passed successfully.
Section 4 of this test covers exactly this code path, so the test would have
caught the problem if it had been run on Windows.

The augmentation arm was then run separately. It completed 5000 rounds in
184 minutes. Running all six arms in parallel took 287 minutes, meaning that
running six jobs at the same time reduced the speed of each individual run
by roughly half.

The result of the separate run is included in the table above and does not
change the overall conclusion.

The failed one-minute run is kept as:

`agent_code/dqn_archive_20260916-014521_augment/`

### Tooling

`tools/sweep.py` gained a `task4` preset (classic, three rule-based opponents,
full action space, score profile) beside the frozen `task1` one that produced
`sweep_results.csv`. It strips ambient `DQN_*` before each run, archives rather
than deletes existing run directories, writes a manifest with the commit hash
and every run's environment before training starts, and gives each run its own
`--log-dir` (six parallel runs all defaulted to
`logs/game.log` and the lines interleaved mid-write, which would have made a
crashed run undiagnosable). `tools/compare.py` now reports score, margin,
suicide and kill rate through `evaluate3.py`, ranks on the pooled tail, prints
the control seed spread, and writes `sweep_task4_results.csv` without touching
the tracked task-1 results. It also merges into that file instead of rewriting
it: opening in "w" and writing only the current run's configurations meant a
`--only` rerun of one arm silently deleted every other arm's rows.

## 2026-09-15 - Targeted opponent-bomb fine-tune prepared

The r1250 diagnosis is specific: crowded suicide fell to 7.8%, but the policy
still drops 44.2 bombs per round, scores only 0.186 kills per round and finishes
strictly first in 25.9% of rounds. More identical `score` training is not
justified because r1500 already regressed. The missing signal is not generic
bombing; it is which bombs can turn into opponent score.

Added `OPPONENT_THREATENED`, emitted only when a newly dropped bomb is escapable
and its current blast line contains at least one opponent. Added a new
`DQN_REWARD_PROFILE=score_hunt` that is the old score reward table plus +0.5 for
that event. The eventual kill still pays +5.0. `GOOD_BOMB` remains zero, so
crate-only demolition remains neutral. The historical `curriculum` and `score`
profiles are unchanged, preserving every previous run's recipe.

`training_log.csv` gains `opponent_threat_bombs`, a subset of `good_bombs`, so
the run can prove whether this dense signal actually occurred. The existing
header-protection logic will archive the previous training log automatically
when the new run begins. The frozen checkpoint table also gains strict-first
and tied-first rates, both on screen and in `eval_curve.csv`, because mean score
alone hid the behaviour the user saw in the GUI.

Focused tests cover an escapable opponent bomb, crate-only bomb, stone-blocked
opponent, and an inescapable aimed bomb. All pass: the suicidal case receives
no hunting bonus; the old `score` profile remains neutral; and `score_hunt`
pays exactly +0.5 before the existing step cost. All augmentation checks still
pass. A two-round frozen smoke run verified the strict-first/tied-first runtime
path. N=2 is only a plumbing check and is not performance evidence. The first
headless reward-test attempt stopped during pygame initialisation; the same test
with dummy SDL video/audio drivers executed every assertion and passed.

No model file was copied, no checkpoint was selected, and no training or long
evaluation was started. Keep `models/task4-score-r1250-ego13.pt` untouched as
the confirmed fallback. The prepared run is:

```bash
cp models/task4-score-r1250-ego13.pt agent_code/dqn_agent/dqn-model.pt

DQN_REWARD_PROFILE=score_hunt \
DQN_LR=0.0001 \
DQN_EPS_START=0.08 \
DQN_EPS_END=0.03 \
DQN_EPS_DECAY=750 \
DQN_AUGMENT=0 \
DQN_SEED=131 \
./venv/bin/python tools/train_curve.py \
  --rounds 1500 \
  --checkpoint-every 250 \
  --eval-rounds 40 \
  --scenario classic \
  --opponents rule_based_agent rule_based_agent rule_based_agent \
  --resume
```

This is a cautious fine-tune, not a promise of improvement. Shortlist only a
checkpoint that improves the r1250 balance: frozen score 3.437 +/- 0.158,
strict-first 25.9%, kill-in-round 17.9%, and suicide 7.8%. A 40-round curve is
selection evidence only; confirm the chosen checkpoint at N=1000 on a new seed
and re-measure the solo and task-3 gates before replacing the fallback.

## 2026-09-15 - Checkpoint curve now displays frozen score

Updated `tools/train_curve.py` so every opponent-evaluation checkpoint line now
shows the DQN's actual frozen mean score and 95% confidence interval before the
existing coins, suicide, kill-in-round and margin metrics. The same values are
appended to `eval_curve.csv` as `frozen_score` and `score_ci95`; all existing
column names and their order are preserved for compatibility with prior curve
readers. Solo checkpoint output is unchanged because, without opponents, score
equals collected coins.

This is evaluation display/bookkeeping only. No reward, model, action or
training behaviour changed, and no training or long evaluation was started.
Static syntax and schema checks passed. A disposable two-round frozen runtime
check then returned both `score` and `score_ci` and rendered the new line format
successfully. Its first two attempts were blocked only because the sandbox did
not permit the framework's normal game/opponent log writes; granting those
specific log paths allowed the same check to pass. The two rounds are a smoke
test and are not reported as performance evidence.

## 2026-09-14 - Repository prepared for the 15.09 team comparison

Fetched `origin` successfully. The latest remote Q-learning branch remains
`origin/agent_code/q_agent` at `c0f36d7` (`add rewards + training log`); the
local Q branch is six commits behind it. The current DQN branch has substantial
unstaged work, so no checkout or merge was attempted.

Exported only `agent_code/q_agent/` from remote commit `c0f36d7` into the
working tree without staging. All six tracked files hash exactly to that commit.
The tracked `model.npy` has six action weight vectors with 34 weights each,
matching the current callbacks and feature count. A one-round frozen launch
against three rule-based agents completed without a crash. That round is a
smoke test only and is not a performance result.

Copied confirmed score-profile r1250 byte-for-byte from the ignored checkpoint
directory to stable archive `models/task4-score-r1250-ego13.pt`. Meeting tools
point at the archive, not the working checkpoint. The live `dqn-model.pt`
remains r1500 and was not changed. `/models` is ignored, so this new archive is
also intentionally unstaged and will require `git add -f` if the team selects it.

Added `tools/compare_team_agents.py`. It evaluates Q and confirmed DQN r1250
separately, frozen, against three rule-based agents under one protocol. It
removes ambient `DQN_*` variables, points directly at the archived r1250 file,
never copies either live model, preserves raw per-round JSON, and writes a
machine-readable summary. Reported metrics are mean score with 95% CI, mean and
individual opponent margins, strict-first/tied-first/behind-top rates, suicide,
kill-round, bombs, invalid actions, and the gate-4 decision.

For fairer separate-process comparisons, `tools/evaluate3.py --seed` now seeds
Python and NumPy agent RNGs as well as the world RNG already seeded by
`WorldArgs`. Candidate actions can still make later trajectories diverge, but
both evaluations begin from the same random streams.

Added three VS Code launch configurations:

- `Meeting: DQN r1250 (GUI, 3 rule based, seed 211)`;
- `Meeting: Q agent (GUI, 3 rule based, seed 211)`;
- `Meeting: compare DQN r1250 and Q (200 frozen rounds)`.

Created `experiments/MEETING_2026-09-15.md` with the current evidence, exact
comparison command, decisions required, deadlines, repository state and red
flags. In particular, the Q README's stored task-2 result is N=150 and its
task-3/4 narrative is stale relative to later commits, so the Q owner must
confirm which checkpoint `model.npy` represents before the result is used.

Verification: launch JSON parsed and all three configurations exist; both
Python files parse; the Q snapshot and 34-feature model match the remote commit;
the Q frozen smoke launch passed; and the combined comparison completed two
disposable rounds for both agents. The first disposable comparison exposed a
path-display error only after calculations finished; the error was fixed and a
second run passed end to end. Both two-round runs wrote only to `/tmp` and are
not evidence. No N=200 comparison, training run, merge, commit or staging was
performed.

## 2026-09-14 - r1250 confirmed: safer and passes the mean gate, not dominant

The user completed the required N=1000 frozen confirmation on unseen world
seed 107. Raw per-round data is
`experiments/runs/2026-09-14-score-r1250-tournament-n1000.json`; it contains
exactly 1000 rows and the expected three independently named rule-based
opponents.

| metric | score-profile r1250 |
|---|---:|
| DQN score | 3.437 +/- 0.158 |
| coins | 2.507 +/- 0.087 |
| kills | 0.186 +/- 0.025 |
| suicide | 7.80% (78/1000) |
| bombs | 44.199 +/- 0.872 |
| invalid | 0.328 +/- 0.035 |
| margin vs RB 0 | +0.560 +/- 0.233, CI [+0.327, +0.793] |
| margin vs RB 1 | +0.534 +/- 0.235, CI [+0.299, +0.769] |
| margin vs RB 2 | +0.559 +/- 0.229, CI [+0.330, +0.788] |

**The assignment's mean-score gate passes.** Every paired 95% interval is
strictly above zero. The safety improvement also holds at useful sample size:
7.8% crowded suicide versus 17.4% for A and 32.5% for C, although those older
models were measured on a different world seed.

The user's visual concern is also correct. Using the 1000 saved per-round rows,
r1250 finished strictly ahead of all three opponents in 259 rounds (25.9%),
tied for top in 85 (8.5%), and was behind at least one opponent in 656 (65.6%).
Across the 3000 individual DQN-vs-RB comparisons it was ahead 1483 times
(49.43%), tied 346 (11.53%), and behind 1171 (39.03%). A positive average
margin therefore does not mean it usually appears as the GUI's sole winner.

Mechanistically, the score reward taught survival more strongly than restraint.
Suicide dropped sharply, but bombing remained high at 44.2 per round and kills
fell to 0.186 per round (a kill occurred in 179/1000 rounds). r1250 is a safer
policy with a valid positive mean margin, not the consistently dominant policy
the user hoped to see. Do not continue the identical recipe: r1500 already
regressed in the 40-round curve, and more of the same has no measured upward
trend.

## 2026-09-14 - Score-aligned fine-tune completed; r1250 shortlisted

The user ran the prepared training command from A r3750: 1500 continuous
`classic` rounds against three `rule_based_agent`s, score reward profile,
`DQN_LR=2e-4`, epsilon 0.10 -> 0.05 over 750 rounds, augmentation off, agent
seed 91. Wall clock was 14.6 minutes. Six older checkpoints were moved safely
to `checkpoints/archive-20260914-165646/` before the run.

Frozen curve, 40 rounds per checkpoint:

| rounds | coins | suicide | kill-in-round | mean margin |
|---:|---:|---:|---:|---:|
| 250 | 2.40 +/- 0.37 | 15.0% | 17.5% | -0.26 +/- 0.92 |
| 500 | 2.20 +/- 0.35 | 10.0% | 10.0% | -0.43 +/- 0.78 |
| 750 | 1.48 +/- 0.39 | 5.0% | 25.0% | -0.39 +/- 0.97 |
| 1000 | 2.10 +/- 0.41 | 17.5% | 5.0% | -0.70 +/- 0.73 |
| **1250** | **3.00 +/- 0.47** | **2.5%** | 12.5% | **+1.42 +/- 1.03** |
| 1500 | 2.33 +/- 0.49 | 2.5% | 10.0% | +0.10 +/- 1.10 |

r1250 is the only candidate. The entire 40-round interval is above zero, but
the interval is wide and the r1500 regression shows that the result is not a
stable upward curve. Per project rule, this is a shortlist only, not a reported
performance claim; because it is a noisy near-threshold selection, confirm at
N=1000 rather than N=200.

Training health checks passed on the final 1500 rows of `training_log.csv`:
epsilon began at 0.0999, reached 0.05 at round 750 and stayed there; buffer grew
continuously to 50,000; mean loss fell by block from 0.1164 to 0.0489; there was
no restart. Training score rose from 1.28 in rounds 1-250 to 2.14 in rounds
1251-1500, but that online number is diagnostic only. With 26 crates and 12.7
good bombs per round in the last block, reward still averaged -7.48, which is
consistent with demolition bonuses being removed rather than the old profile
paying at least +25 from those two counters alone.

Operational state: `agent_code/dqn_agent/dqn-model.pt` is the newly trained
r1500 working model (SHA-256
`7151d1ad7f8aa779cca8029ccda8e72e0ad21fea9b41d1f72ce21eee204d0bb1`).
The r1250 candidate is
`agent_code/dqn_agent/checkpoints/dqn-r001250.pt` (SHA-256
`a09e1c3e295ff7085ae211a09e084c328951b8793b8b7a727f8d7bb9f5ea82fa`).
Measured fallback C remains unchanged in `models/` with SHA-256
`23b506a90b2bdf8d725863bde9ec3f9bbf4b0dccee551b7509cd88a975ee3b11`.

## 2026-09-14 - Both finalists look weak; score-aligned retrain prepared

The user's same-board visual comparison showed both C and A losing. This does
not invalidate the 1000-round mean-score measurements, but it exposes what that
gate does not say: a positive average margin is not a high probability of
winning any one round. The frozen matrix already shows unstable behaviour. C
has 32.50% crowded suicide and kills in 26.3% of rounds; A has 17.40% crowded
suicide and kills in 27.3%.

The learning curves do not justify simply running the old setup longer. A's
40-round margin jumps between negative and positive values through the whole
run, with no stable upward trend; C's six points are similarly flat. The chosen
checkpoints are local noisy peaks, not the end of a converging curve.

The reward mismatch is large enough to explain the visible policy. In the
tournament matrix C scores 3.79 actual points but destroys 48.10 crates per
round. `CRATE_DESTROYED=+0.5` therefore contributes roughly +24 training reward
on its own, before `GOOD_BOMB=+1` and `COIN_FOUND=+0.5`. The learner is paid much
more to demolish the board than to win, and C drops 38.53 bombs per crowded
round against roughly 18 from each rule-based opponent.

Added one isolated training option: `DQN_REWARD_PROFILE=score`. Its positive
task-progress event rewards are the real scoreboard events,
`COIN_COLLECTED=+1` and `KILLED_OPPONENT=+5`; crate, coin-found and good-bomb
bonuses become zero. Existing safety shaping remains, with
`SUICIDAL_BOMB=-5` and both death events at -10. The default is still
`curriculum`, so every historical result remains reproducible and the installed
frozen model behaves exactly as before.

Verification passed: the file parses; the legacy profile still gives +1.98 for
one crate + one found coin + one good bomb including the step cost; the score
profile gives -0.02 for the same events; a suicide's two death events give
-20.02; profile selection works; and all augmentation checks still pass. The
first sandboxed test attempt was blocked from writing Python bytecode and
pygame could not initialise audio/video; rerunning without bytecode and with
dummy SDL drivers passed. No training or evaluation run was started.

## 2026-09-14 - Side-by-side C and A visual launchers

Added `Compare C r1000 (GUI, 3 rule based, seed 77)` and
`Compare A r3750 (GUI, 3 rule based, seed 77)` to VS Code Run and Debug. Both
use the same `classic` world seed, three rule-based opponents, one frozen round,
the project virtual environment, and tournament-default play settings. The only
intentional environment difference is the absolute `DQN_MODEL_FILE`, pointing
at the corresponding archived checkpoint. All other play-time `DQN_*`
variables are removed. Launching either configuration does not copy or modify
`agent_code/dqn_agent/dqn-model.pt`.

This is for behavioural inspection only. One seeded GUI round is not a
performance measurement and does not supersede the 1000-round frozen matrix.
No game or training run was started while adding the configurations.

## 2026-09-14 - The bad-looking GUI round loaded the right model

Investigated the visual run against three `rule_based_agent`s before considering
another training run. The installed `dqn-model.pt` still has SHA-256
`23b506a90b2bdf8d725863bde9ec3f9bbf4b0dccee551b7509cd88a975ee3b11`,
identical to `models/task4-C-solo-refresh-r1000-ego13.pt`. The agent log says
`Loaded model from dqn-model.pt (view ego:13)`, and the game log shows no agent
error or decision-time violation. The launch configuration has no `--train`
flag and explicitly removes all play-time `DQN_*` overrides.

The game log ends during the round, with no `WRAPPING UP ROUND` or `SHUT DOWN`,
so the watched game was stopped before completion. It cannot be treated as a
result, and a single completed game would still be only a visual smoke test.

The apparent weakness is consistent with the measured policy rather than a load
failure. C's 1000-round frozen DQN score is 3.79 +/- 0.19, and its weakest
paired margin against the three rule-based opponents is only +0.54 +/- 0.28.
Its crowded suicide rate is 32.50% with a 95% upper bound of 35.47%. It wins the
required comparison in expectation, but it is volatile and can look plainly
worse in an individual game.

**Decision: do not retrain from one incomplete GUI run.** If the objective is a
calmer, safer DQN rather than the best cross-task DQN, the already measured A
r3750 checkpoint is the controlled alternative: DQN score 3.81 +/- 0.19,
weakest rule-based margin +0.63 +/- 0.27, and crowded suicide 17.40% with a 95%
upper bound of 19.87%. Switching checkpoints is cheaper and better evidenced
than launching an unpriced run. Any new training proposal must first define the
target metric and reward stream, change one thing, and then re-measure every
gate frozen.

## 2026-09-13 - One-click frozen play environments

Added two VS Code Run and Debug configurations for visually inspecting the
selected DQN: one against three `random_agent`s and one against three
`rule_based_agent`s. Each opens the GUI for one round, uses `venv/bin/python`,
does not pass `--train`, and removes all nine `DQN_*` variables read by
`callbacks.py`. The existing headless submission-check configuration was
hardened with the same interpreter and clean environment. No game or training
run was started by this edit.

## 2026-09-13 - Assignment audit: DQN done, whole project not submitted

Read all 12 pages of `final_project.pdf` and checked the repository against the
actual submission rules. This corrects an important ambiguity in the handoff:
17.09 at 21:00 is the optional official pre-run/crash-test deadline; the final
agent-code deadline is 21.09 at 21:00, and the report is due 28.09 at 21:00.

### What is genuinely finished

The DQN arm is finished. C r1000 is selected within that arm, installed,
byte-identical to its archive, measured frozen across every outstanding gate,
and passes the one-round stock-framework submission shape with `train=False`
and three `random_agent`s. Its play-time code uses CPU, pins PyTorch to one
thread, has no multiprocessing, loads its weights from its own directory, and
does not depend on the training callbacks during official play. No further DQN
training is justified unless the official pre-run exposes a compatibility bug.

### What still blocks calling the whole project finished

1. The assignment requires the best model across the team to be submitted.
   The project handoff says the feature-based Q-learning model is the intended
   tournament entry, but the `agent_code/q_agent` branch visible in this clone
   contains only an early implementation and no tracked trained parameters.
   That may be stale relative to the teammates' machines, so their current
   checkpoint must be obtained, measured, and compared before packaging.
2. The current DQN changes are not committed or pushed. The final C archive and
   two other task-4 archives remain ignored, and the configured GitHub URL
   returns 404 to an unauthenticated visitor, so the public-repository
   requirement is not yet demonstrably met.
3. No submission zip exists. The live `agent_code/dqn_agent` directory is about
   109 MB because it includes checkpoints, logs, CSVs, and `__pycache__`; it
   must not be zipped wholesale. The final zip should contain only the winning
   agent directory, its runtime source, and its trained parameters, plus a
   `requirements.txt` only if that agent needs a library absent from the
   supplied Docker image.
4. The native stock-settings crash check passed, but the exact clean submission
   payload has not been run inside the supplied Docker image. Docker is
   installed on this Mac, but this session could not access the daemon socket,
   so that verification remains for the owner to run.
5. The report still remains. For three team members the requested length is
   about 12,000 words total and not much more. Required sections are
   Introduction, Background, Project planning, Methods, Training, Experiments
   and Results, and Conclusion. Every subsection needs a named responsible
   author; the report must include the public repository URL, dependencies,
   both learning approaches, systematic comparisons and negative results, and
   must not be committed to the public repository or use the university logo.
   AI-assisted drafts must be rewritten in the team's own style, with the main
   work clearly attributable to the team. The planning section must also show
   real collaboration across the two models rather than isolated model silos.

The official upstream `master` revision checked on 12.09 is already contained
in this repository's history. No missing framework update was found. The next
project action is team-wide model reconciliation, not more DQN training.

## 2026-09-12 - Submission check passes with the selected model

Re-ran the official one-round crash/configuration shape with C installed:
stock `settings.py`, no `DQN_*` variables, `train=False`, `classic`, and three
`random_agent` opponents. To make the stock-file condition literal without
touching the working tree, the check ran from a temporary archive whose
`settings.py` was byte-identical to `origin/master`; the selected
`dqn-model.pt` was byte-identical to the working copy and C archive.

**PASS: exit code 0, round completed, no crashes, agent errors, or decision-time
violations.** The agent loaded `dqn-model.pt` with checkpoint view `ego:13`.
This is deliberately only a pass/fail submission check; one round is not a
performance measurement.

The sandbox has no accessible macOS audio/display device, so an initial harness
launch aborted inside `pygame.init()` before `settings.py` or the agent was
loaded. The successful run used SDL's dummy audio and video drivers. Those are
framework display settings, not `DQN_*` agent settings, and the game remained
headless. The temporary checkout was removed after verification.

## 2026-09-12 - Final candidate matrix measured; C selected and installed

All eight missing measurements completed frozen. The command ran each process
with a clean environment and an absolute `DQN_MODEL_FILE`; `evaluate3.py`
constructed every agent with `train=False`. Raw terminal output is in
`experiments/runs/2026-09-12-candidate-final-evaluation.txt`.

### Gate 4, 1000 rounds, us plus three `rule_based_agent`s

| model | DQN score | margin vs RB 0 | margin vs RB 1 | margin vs RB 2 | suicide | invalid |
|---|---|---|---|---|---|---|
| **C r1000** | 3.79 +/- 0.19 | +0.63 +/- 0.26 | +0.62 +/- 0.26 | +0.54 +/- 0.28 | 32.50% (95% upper 35.47%) | 2.25 +/- 0.12 |
| A r3750 | 3.81 +/- 0.19 | +0.77 +/- 0.26 | +0.63 +/- 0.27 | +0.68 +/- 0.26 | 17.40% (95% upper 19.87%) | 0.75 +/- 0.06 |

Both pass: even the weakest paired 95% interval is wholly above zero. A's
safety advantage is real, but the two DQN score intervals are effectively
identical.

### Task 2, solo `classic`, 1000 rounds

| model | coins | suicide | invalid |
|---|---|---|---|
| **C r1000** | **7.70 +/- 0.11** | 1.00% (95% upper 1.83%) PASS | 0.00 +/- 0.00 |
| A r3750 | 6.33 +/- 0.13 | 0.50% (95% upper 1.17%) PASS | 0.00 +/- 0.00 |

Both pass survival and fail the 8-coin gate. C's whole coin interval is below
8, so the preliminary 7.87 +/- 0.19 at 300 rounds was not a pass and did not
hold as the final estimate.

### Task 3

| model | kills `peaceful` in a round (300r) | margin vs `coin_collector` (400r) |
|---|---|---|
| **C r1000** | **76.7% [95% 71.6%, 81.1%] FAIL** | **+0.13 +/- 0.40 FAIL** |
| A r3750 | 59.3% [95% 53.9%, 64.7%] FAIL | -0.82 +/- 0.38 FAIL |

Neither gate passes. C repaired most of the specialist's loss against
`peaceful_agent` and is close in point estimate, but its 95% lower bound is well
below 80%, so there is no pass to confirm. Against `coin_collector_agent`, C is
indistinguishable from a tie and A is clearly behind.

### Shipping decision

**C r1000 ships.** Gate 4 is the specification's required gate and both models
pass it with effectively tied DQN scores. C is then clearly better on solo coins
and both task-3 measures. A's lower suicide and invalid-action rates do not turn
into a score advantage. No further training is justified before the deadline.

Copied `models/task4-C-solo-refresh-r1000-ego13.pt` to
`agent_code/dqn_agent/dqn-model.pt` and verified the files byte-for-byte. Both
have SHA-256
`23b506a90b2bdf8d725863bde9ec3f9bbf4b0dccee551b7509cd88a975ee3b11`.
The next action is the clean submission test.

## 2026-09-12 - Evaluation preflight and housekeeping

No model number was produced and no long evaluation was started. Both candidate
paths resolve, `tools/evaluate3.py` still hard-codes training off, and the shell
has no ambient `DQN_*` variables. The ignored `tools/.train_curve.py.bak`
leftover was deleted. The next action remains the eight frozen measurements in
the STATUS block, run on the M3 Max.

## 2026-09-12 - Two ways out of the specialist trap, and the cheap one won

Two runs, both resumed from `task4-classic-r3250-ego13.pt`, the specialist that
passes gate 4 but collects 4.95 coins solo against r3750's 7.65.

| | what it changed | rounds | wall clock |
|---|---|---|---|
| **A** | opponents: `rule_based`, `rule_based`, `peaceful` instead of 3x `rule_based` | 4000 | 21.4 min |
| **C** | solo board, `DQN_LR=2e-4`, `DQN_EPS_START=0.1` | 1500 | **7.6 min** |

Both curves measured against three `rule_based_agent`s regardless of what they
trained on, using the new `--eval-opponents`. That flag exists because of
2026-09-11: measuring only what you trained against is how a specialist gets
mistaken for a better agent.

### The diagnosis that pointed at both runs

Solo, 300 frozen rounds, the specialist against its parent:

| | coins | crates | **bombs** | suicide |
|---|---|---|---|---|
| r3750 | 7.65 | 114.7 | **47.6** | 1.00% |
| r3250 specialist | 4.95 | 90.8 | **26.5** | 0.00% |

It was not lost, looping or dying. It dropped barely half as many bombs. Four
thousand rounds of being punished by `rule_based_agent`s taught it caution, and
solo that caution is pure loss: a bomb not dropped is a coin not collected. It
had the information to tell the situations apart, channel 3, and no reason to
learn the distinction because it never saw an empty board.

### Confirmed results

Solo at 300 rounds, tournament shape at 400 unless noted:

| model | solo coins | solo bombs | task-4 margin | suicide (crowded) | invalid |
|---|---|---|---|---|---|
| r3750 all-rounder | **7.65** +/- 0.24 | 47.6 | -1.02 +/- 0.40 (200r) fail | 42.50% | 1.00 |
| r3250 specialist | 4.95 +/- 0.31 | 26.5 | +0.78 +/- 0.25 (1000r) PASS | 33.20% | 2.35 |
| **A r3750** | 6.49 +/- 0.24 | 31.3 | +0.58 +/- 0.34 PASS | **15.25%** | **0.74** |
| A r2750 | 5.18 +/- 0.29 | 27.2 | +0.44 +/- 0.35 PASS | 13.25% | 0.71 |
| **C r1000** | **7.87** +/- 0.19 | 46.2 | +0.66 +/- 0.38 PASS | 31.75% | 2.15 |

**All three new checkpoints pass gate 4**, and the bombing-rate column tracks
the solo coins almost exactly, which is the diagnosis confirming itself.

**C is the result.** 7.6 minutes of solo training at a fifth of the learning
rate recovered solo coins to 7.87, *above* the 7.65 it started life at and the
best number this project has produced, while keeping the task-4 margin at +0.66
with the whole interval above zero. I predicted the opposite: "standard result
is that it swings back and loses the gate". It did neither. Best value per
minute of any run in this log.

**A did a different job.** Mixed opponents recovered solo coins only partly,
4.95 to 6.49, but halved self-kills in the crowded game, 33.2% to 15.25%, and
cut invalid actions from 2.35 to 0.74 per round, the cleanest of any model in
that shape.

The two are not ranked, they are different trades. C wins on score everywhere.
A is markedly more disciplined. C got its score back by bombing more (46.2
bombs against A's 31.3), and more bombing in a crowded room is more ways to
die, which is exactly where its 31.75% comes from. If the report leads with
score, C. If it leads with the survival story the spec emphasises, A.

### Queued, then stopped unfinished on 2026-09-12

None of the following has a number. The measurements were started and stopped
before any of them completed, so there is nothing to quote and nothing to
recover; they have to be run again. The commands are in the STATUS block.

- C r1000, task-4 margin at **1000** rounds on seed 77 (rule 4: re-measure near
  a threshold before claiming a pass).
- C r1000, **solo at 1000 rounds**. At 7.87 +/- 0.19 the interval is
  [7.68, 8.06] and the task-2 coin gate is 8, so the point estimate cannot pass,
  but this is the number that goes in the report and it deserves n=1000.
- C r1000 and A r3750 against `peaceful_agent` and `coin_collector_agent`, the
  two task-3 gates. Both regressed badly for the specialist (69.2% -> 37.7%
  kill-in-round) and it is an open question whether either run repaired that.

### Operational note: the device file transfer is not trustworthy

Twice now, writing a tool file to the Mac reported success and left the old
content in place, and the second time it silently **reverted** a fix that had
been verified present an hour earlier. The `max()` versus mean margin fix was
lost that way. Both times the reliable route was to edit the file on the Mac
directly and then read the patched lines back off disk.

Rule from it, on top of 2026-09-11's "every scripted edit gets an assertion":
**verify by reading the file that will actually run, after it has been written.**
Not the copy that was sent, and not the tool that shares the bug.

## 2026-09-11 - Gate 4 passes, and it cost us task 2

4000 rounds on `classic` against three `rule_based_agent`s, resumed from r3750,
`DQN_EPS_START=0.3`, `DQN_EPS_DECAY=2000`. 22.7 minutes. One variable from the
task-3 run: the opponent.

### The gate

`models/task4-classic-r3250-ego13.pt`, frozen, in the tournament shape:

| | rounds | seed | our score | `rule_based` | margin |
|---|---|---|---|---|---|
| shortlist from the curve | 40 | - | | | +1.23 +/- 1.08 |
| confirmation | 400 | 41 | 4.24 | 3.25 | **+0.99** +/- 0.40 |
| re-confirmation | **1000** | **77** | 4.08 | 3.30 | **+0.78** +/- 0.25 |

**Gate 4 PASSED.** The whole interval, [+0.53, +1.03], is above zero at 1000
rounds on a seed the model was not shortlisted on. Baseline r3750 in the same
shape is -1.02 +/- 0.40.

That is the first gate this arm has passed since task 1, and the only one the
spec actually requires.

### It is a specialist, and that is the real finding

Same weights, everything else we measure:

| | r3750 (task-2 weights) | **task-4 r3250** |
|---|---|---|
| solo `classic`, coins (task-2 gate) | **7.51** | **4.34** +/- 0.38 |
| solo `classic`, suicide | 0.60% | **0.50%** |
| solo `classic`, invalid actions | 0 | 0 |
| tournament shape, margin | -1.02 +/- 0.40 | **+0.78** +/- 0.25 |
| 1v1 vs `rule_based`, margin | -1.85 +/- 0.50 | -0.89 +/- 0.45 |
| vs `coin_collector`, margin | **+0.17** +/- 0.47 | -0.25 +/- 0.43 |
| vs `peaceful`, kill-in-round | **69.2%** | 37.7% |
| vs `peaceful`, margin | **+10.80** | +6.58 |

It beats three `rule_based_agent`s and is worse at everything else, including
the base game it was built on. Solo coin collection fell from 7.51 of 9 to 4.34.
Both task-3 numbers went backwards. This is opponent-distribution overfitting,
measured cleanly, with the before and after on seven different boards. For a
report whose whole point is comparing two learning approaches, that is a better
result than a model that is quietly good at one thing.

Bomb safety did **not** degrade: solo suicide 0.50% against 0.60%, and 33.2%
against 42.5% in the crowded game. The agent got safer and narrower at once.

### My prediction was wrong, and here is the part I got wrong

Written before the run: margin around -0.4 to -0.6, self-kills down, coins down,
"an improvement that still does not pass", and that anything much better needs
explaining. The margin came out **+0.78**.

What I got right: deaths down (42.5% to 33.2%), coins roughly flat in that
shape, avoidance being worth points here.

What I got wrong, and why. I priced the kill signal at the **baseline** rate of
0.115 kills per round and treated it as fixed. It was not fixed: during training
it climbed from 0.010 to 0.147 per round, and frozen it reached 0.366. The
margin gained almost entirely through kills, exactly the channel I had written
off.

| run | total kills over 4000 rounds | deaths per kill | kills/round, first -> last block |
|---|---|---|---|
| task 3, vs `coin_collector` | 81 | 46 to 1 | 0.003 -> 0.025, flat |
| task 4, vs 3x `rule_based` | **258** | 14.7 to 1 | 0.010 -> **0.147**, rising |

So the refinement to this morning's lesson: **signal density is a property of the
opponent's behaviour, not just of our policy.** `rule_based_agent` walks toward
opponents and bombs them, so it puts itself inside blasts we are already
setting off. `coin_collector_agent` ignores us entirely and farms crates
somewhere else. An opponent that comes to you is a dense teacher, and one that
ignores you cannot teach you to fight no matter how long you train.

The other half of the miss is plainer: I priced the gain and never priced the
cost. Nothing in my prediction asked what 4000 rounds of three-opponent boards
would do to solo coin collection, and the answer was -3.17 coins.

### One number worth watching

Invalid actions per round, which have been 0.00 all project:

| | ours | `rule_based`'s |
|---|---|---|
| solo | 0.000 | - |
| tournament shape, r3750 | 1.005 | 6.87 |
| tournament shape, r3250 | 2.348 | 6.92 |

Not a defect introduced by training: in a crowded game an opponent can step into
the tile we chose after `legal_actions` judged it free, and the reference agent
eats 6.9 of these per round to our 2.3. Still, ours doubled, and gate 1's
standard was zero. Worth a sentence in the report rather than silence.

### Decision this leaves

Two models, a clean trade-off, and 6 days:

| | r3750 | r3250 |
|---|---|---|
| passes | task 1, task-2 survival | **task 4** |
| fails | task 4, task-3 margin | task-2 coins, both task-3 numbers |

`agent_code/dqn_agent/dqn-model.pt` currently holds the task-4 r4000 weights,
which are neither of these. Whatever ships has to be copied in and re-tested
against stock settings before 17.09.

## 2026-09-10 (night) - Task 4 baseline, and the margin taken apart

`models/task2-classic-r3750-ego13.pt`, frozen, mask on, no task-4 training. Two
setups, because the gate does not say which and they disagree about *why* we
lose.

| | our score | `rule_based` | margin | our suicide | they killed us | our kills |
|---|---|---|---|---|---|---|
| 1v1, 400 rounds | 3.99 | 5.84 | **-1.85** +/- 0.50 | **38.25%** | 17.5% of rounds | 4.2% of rounds |
| us + 3 `rule_based`, 200 rounds | 2.56 | 3.58 (mean of 3) | **-1.02** +/- 0.40 | **42.50%** | 74.5% of rounds | 11.5% of rounds |

Neither passes. The suicide rate completes a pattern that has held all day: the
better the opponent, the more the agent kills itself.

| opponent | our suicide rate |
|---|---|
| nobody (solo `classic`) | 0.60% |
| `peaceful_agent` | 3.50% |
| `coin_collector_agent` | 13.25% |
| `rule_based_agent` | **38.25%** |

### Where the margin actually comes from

Score is `coins + 5 x kills`, so the margin splits cleanly:

| | margin from coins | margin from kills | total |
|---|---|---|---|
| 1v1 | **-1.19** | -0.66 | -1.85 |
| tournament shape | -0.35 | **-0.67** | -1.02 |

Two different problems wearing the same number. One against one we are simply
out-collected, which is the task-2 gate we never passed showing through: solo we
get 7.51 coins on `classic` and `rule_based_agent` gets 8.60. In the tournament
shape the coin gap nearly vanishes, because nine coins split four ways leaves
little to be better at, and almost the whole deficit is that **they kill us and
we do not kill them**. Every death we hand over is worth 5 points to somebody.

### And unlike task 3, suicide is not the whole story

Splitting the rounds the same way as this morning:

| | 1v1 margin | tournament margin |
|---|---|---|
| rounds we survived | -1.49 | -1.03 |
| rounds we self-killed | -2.42 | -1.01 |
| **if the self-kill rounds went like the survivors** | **-1.49, still fails** | **-1.03, still fails** |

Against `coin_collector` the whole deficit was suicides. Against
`rule_based_agent` it is not: even a perfect-survival version of this agent
loses. That is a capability gap, not a safety gap.

### Masking is exhausted

Re-priced the escape-check candidates against `rule_based_agent`, 120 rounds, 48
fatal bombs. Every candidate blocked **0 of 48**, and 0 of 4298 survived bombs.

That is not a negative result about the idea; the play-time opponent-aware mask
is already switched on in these games, so the bombs it would catch never reach
the sample. What it says is that the remaining 38% of self-kills all pass a check
that knows where the opponents are and where they can walk. The escape exists
when the bomb is dropped and the policy fails to walk it. No masking change fixes
that. Training does, or nothing does.

### The prediction for the next run, written down before it is spent

Applying this morning's lesson: price the reward *stream*, not the reward table.
In the tournament shape the agent takes roughly 1.2 deaths per round and scores
0.115 kills, so the opponent-related signal runs about **10 to 1 toward
avoidance**. That is the same shape that made the task-3 run go backwards.

The difference, and it is the reason this run is still worth spending: **for
task 4 avoidance is worth points.** The gate is a score margin, every death we
avoid denies an opponent 5 points, and -0.67 of the -1.02 is exactly that. In
task 3 the gate wanted aggression and the reward taught caution, so the two
fought. Here they point the same way.

So the prediction, to be checked rather than reinterpreted afterwards: self-kills
fall substantially, our coins fall somewhat, the kill component of the margin
improves by most of 0.67, and the total margin lands somewhere around -0.4 to
-0.6. **That is an improvement that still does not pass.** If it comes out much
better than that, something else is going on and it needs explaining. If it comes
out worse, avoidance training is exhausted as an idea and the honest move is to
stop running and write up.

### The evaluation phase crashed, and it was my bug twice over

`KeyError: 'rule_based_agent'` on the first checkpoint, after 22.7 minutes of
training had already completed. The training is fine and the 16 checkpoints are
on disk; only the curve had to be rebuilt with `--eval-only`.

The cause is the duplicate-renaming described above. I found that problem before
the run, patched both tools, and then **verified the wrong one**: the edit to
`tools/evaluate3.py` had an assertion and applied, the second edit to
`tools/train_curve.py` was a plain `str.replace` with no assertion, its anchor
did not match, and it silently did nothing. I then ran a four-agent game through
`evaluate3.py`, saw it pass, and called both fixed.

Two rules out of it, both of which this project already knew in other forms:

- **Every scripted edit gets an assertion.** A `replace` that matches nothing
  returns the original string and reports success, which is the same failure
  shape as every silent measurement bug in this log.
- **Verify the tool that is about to run**, not a tool that shares the bug.

Both files are now fixed and the fix was confirmed by reading the patched lines
back off disk, not by trusting the write.

While fixing it, a second and more consequential error in the same function:
the margin was computed against `max()` of the opponents' scores. Gate 4 says
"mean score strictly above `rule_based_agent`'s", which means a typical
opponent, not the luckiest one in each round. With three opponents `max()` reads
about 2.5 points harsher, so the curve would have reported roughly -3.8 where
the gate reads -1.0, and every checkpoint would have looked hopeless. It now
compares against the mean, which reproduces the -1.02 baseline. That one would
not have crashed. It would just have been wrong.

### Tooling fix this needed

`tools/evaluate3.py` and `tools/train_curve.py` both indexed results by the agent
names passed on the command line. The framework renames duplicates, so three
`rule_based_agent` entries come back as `rule_based_agent_0`, `_1` and `_2`, and
both tools raise `KeyError` on the first checkpoint of a tournament shaped run.
They now read the keys the rows actually carry. (`evaluate3.py` was fixed before
the run; `train_curve.py` was not, despite my saying it was. See the next
section.)

## 2026-09-10 (evening) - The first task-3 training run, and why it went backwards

4000 rounds on `classic` against `coin_collector_agent`, resumed from r3750,
`DQN_EPS_START=0.3`, `DQN_EPS_DECAY=2000`. One variable changed from the run
that produced r3750: an opponent in the game. No reward change, deliberately, so
that a null result would be interpretable. 19.7 minutes on the M3 Max.

### The training itself worked

| rounds | steps | coins | crates | suicidal bombs | reward | epsilon |
|---|---|---|---|---|---|---|
| 1-1000 | 29 | 0.41 | 14.0 | 0.016 | +2.72 | 0.269 -> |
| 1001-2000 | 79 | 1.24 | 31.2 | 0.015 | +20.75 | |
| 2001-3000 | 146 | 2.28 | 48.0 | 0.048 | +40.97 | 0.050 |
| 3001-4000 | 144 | 2.20 | 47.6 | 0.033 | +40.66 | 0.050 |

Learning curve rose and flattened by round 2500, same shape as every previous
run. Epsilon annealed cleanly, the buffer filled and stayed at 50000, suicidal
bombs stayed near zero. Nothing was broken.

### The frozen result went the wrong way

Two checkpoints shortlisted from the curve and confirmed properly, 400 rounds
against the coin collector and 300 against the peaceful agent, against the r3750
baseline measured the same way:

| model | suicide vs cc | margin vs cc | coins vs cc | suicide vs pf | kill-in-round vs pf |
|---|---|---|---|---|---|
| **r3750** | 13.25% [10.3, 16.9] | **+0.17** +/- 0.47 | **4.80** | 3.50% | **69.2%** |
| task-3 r2500 | **9.25%** [6.8, 12.5] | -0.23 +/- 0.38 | 4.12 | 2.33% | 56.3% [50.7, 61.8] |
| task-3 r3250 | 11.75% [9.0, 15.3] | -0.29 +/- 0.42 | 4.10 | 3.00% | 47.7% [42.1, 53.3] |

Survival improved, and that part is real: 13.25% to 9.25%. Everything the gates
actually measure got worse. The kill rate against the peaceful agent fell from
69.2% to 56.3%, and those intervals do not overlap, so that is a regression and
not noise.

**The agent traded productivity for caution.** It is exactly what you would
build if you were told deaths are expensive and told almost nothing else.

### The arithmetic that explains it, which I should have done first

Across the whole run:

| | count |
|---|---|
| kills scored | **81** |
| rounds ending early, essentially always our death | **3736 of 4000** |
| reward from kills, as a share of total reward | **0.31% to 0.52%**, every block |

One kill per 49 rounds. 46 deaths for every kill. `KILLED_OPPONENT` is +5.0 and
`GOT_KILLED` and `KILLED_SELF` are -5.0 each, so the net opponent-related signal
the agent experienced was overwhelmingly "stay away from that thing". It learned
precisely that, and the measurement above is what learning it looks like.

This also closes the loop on this morning's blinding control, which found that
zeroing channel 3 changes nothing. Of course it does. In 600,000 training steps
the channel was attached to 81 positive events.

Rounds ending early is not incidental either: with `--train 1` the framework
stops the round when the training agent dies, so a policy that dies at step 144
of 400 sees only the first third of every board. 47.6 crates destroyed out of
119 on `classic` confirms these are deaths, not cleared boards.

### What I got wrong

I chose `coin_collector_agent` as the training partner and argued it was the
harder, more transferable opponent. That reasoning was about board difficulty and
ignored signal density, which is the thing that actually decides whether a
behaviour can be learned. Against `peaceful_agent` the same frozen agent kills in
69% of rounds, roughly 35 times denser, because the peaceful agent random-walks
into blasts we are setting off anyway. If the kill behaviour is to be trained in
at all, it has to be trained where kills happen.

The pricing habit that has worked all project is to do the arithmetic before
spending the run. Here I priced the reward *table* (it already had
`KILLED_OPPONENT`) and never priced the reward *stream*. Those are different
questions and only the second one predicts what gets learned.

### The open question this leaves

Tasks 1 to 3 and their thresholds are our own curriculum. The spec requires
task 4: mean score over 200 rounds strictly above `rule_based_agent`, which it
calls the entry ticket for the tournament. That has not been started, and there
are 6 days.

Three options, and this is a decision, not a finding:

1. **Train against `peaceful_agent`** to get the hunting half of gate 3. Same
   command, one word changed. Highest chance of moving a task-3 number, lowest
   value toward the spec.
2. **Go to task 4 now**, against `rule_based_agent`. It is the requirement, it
   is a much denser and more informative opponent than either task-3 agent, and
   whatever it teaches also applies to task 3.
3. **Ship r3750 and spend the remaining days on the report.** The submission dry
   run already passes and the DQN arm is the comparison, not the tournament
   entry. Two clean negative results (augmentation untested, opponent training
   backfired with the arithmetic to explain why) are worth more in a report than
   a rushed fourth run.

## 2026-09-10 - Task 3 measured, and a bug that measured nothing at all

### The bug first, because it invalidated the first hour of the session

**Symptom.** `DQN_MODEL_FILE=models/task2-classic-r3750-ego13.pt python
tools/evaluate.py --agents dqn_agent --scenario classic --n-rounds 30` reported
**0.00 coins, 400.0 steps, 0 bombs**. The identical command with the identical
file given as an *absolute* path reported **7.72 coins**.

**Cause.** `agents.py`, `SequentialAgentBackend.send_event`, wraps *every*
agent callback in an `os.chdir` into `agent_code/<agent>/`. So while our code
runs, including the import of `callbacks.py` where `MODEL_FILE` is built, the
working directory is the agent's own folder, not the repo root.
`models/task2-classic-r3750-ego13.pt` therefore resolved to
`agent_code/dqn_agent/models/task2-classic-r3750-ego13.pt`, which does not
exist, `will_load` came out False, and `setup` handed the game a randomly
initialised network. There is a warning for exactly this case and it is useless:
it goes to the agent's own log file, which was never even created in that run.

The chdir behaviour was already known and written down (2026-08-31, and it is
the reason every path in `callbacks.py` is built from `Path(__file__)`). The one
path that came from an environment variable instead was the one that broke. Any
future switch that names a file needs the same treatment.

**Evidence it was untrained weights and not a policy.**

- The greedy policy shuttles RIGHT, LEFT, RIGHT, LEFT out of the starting corner
  and never leaves, for all 400 steps of all 30 rounds.
- Q values read off the checkpoint by hand say DOWN (9.109) at that tile. The
  agent in the game did not go down. The network in the game was not that
  network.
- With `DQN_PLAY_EPSILON=0.05` on top, the same agent kills itself in **100%**
  of rounds, mean round length 11 steps.

**Fix** (`callbacks.py`, the file that ships): a relative `DQN_MODEL_FILE` is
resolved against the repo root, and a path that does not resolve to an existing
file raises `FileNotFoundError` instead of falling through to random weights.
Verified three ways: the relative path now measures 8.17 coins, a nonexistent
path now crashes the run instead of reporting 0.00, and with no variable set the
agent still loads its own `dqn-model.pt` exactly as before. Official games never
set the variable, so the raise cannot fire in the tournament.

**Sixth time.** Sixth measurement artefact that looked like a modelling result.
The pattern is identical every time: a number that is quietly wrong is worse
than a run that crashes.

**What it cost.** The opponent-perception test below was run first against the
broken agent and "failed", and it produced a confident and completely wrong
conclusion: that the 13x13 crop never sees an opponent, so task 3 needed a
wider view or a hand-crafted compass feature. Both agents were sitting in their
corners because ours was not playing. Measured properly the opposite is true.
Recording this because that wrong conclusion would have cost a retrain.

### Channel 3 works, and the ego crop is not the problem

`tools/test_others.py`, 41 checks, all passing. Channel 3 has been all-zero for
the whole project because tasks 1 and 2 have no opponents, so it was worth
testing before anything was built on it.

- Channel 3 lights exactly one tile per opponent, at the right coordinates, in
  the global view and under the `ego:13` crop, in all four directions.
- The crop's blind spot is real but bounded: an opponent 6 tiles away is
  visible, 7 tiles away is not.
- `steps_until_lethal` already reads every bomb on the board. An opponent's bomb
  produces exactly the same danger map as our own; ownership does not enter it.
- `legal_actions` already blocks a tile an opponent is standing on, and only
  that tile. `escape_exists` treats two opponents sealing a corner as no escape.
- The 8-fold augmentation moves opponents with the board, all eight elements.

Live, against `peaceful_agent` on `classic`, with the weights actually loaded:
the opponent is inside the 13x13 window on **37%** of steps, and over 30 rounds
**28 of 30** bring the two agents within 6 tiles of each other, median closest
approach 2 tiles. Against `coin_collector_agent` it is 48% of steps and 30 of
30. So the agent can see what it needs to see, and no view change is justified.

### The task-3 baseline: same weights, three boards

`tools/evaluate3.py`, 200 rounds each, training off, `models/task2-classic-r3750-ego13.pt`.

| | our score | their score | margin | kills in >=1 round | our suicide |
|---|---|---|---|---|---|
| solo `classic` (1000 rounds, 09.09) | 7.51 coins | | | | **0.60%** |
| vs `peaceful_agent` | 11.40 | 0.15 | **+11.24** +/- 0.52 | **74.5%** [68.0, 80.0] | **6.00%** |
| vs `coin_collector_agent` | 4.40 | 4.94 | **-0.54** +/- 0.66 | 8.0% | **21.50%** |

Neither gate passes. The kill rate against the peaceful agent is 5.5 points
short of 80, and the score margin against the coin collector is a statistical
tie that leans the wrong way.

Two things in that table matter more than the gates themselves.

**The suicide rate scales with the number of agents on the board: 0.60%, 6.0%,
21.5%.** The agent learned bomb safety on an empty board and it does not
transfer. Against the coin collector it kills itself in more than one round in
five, which is ten times the task-2 gate it already passed.

**Coin collection halves against a real opponent**, 7.51 to 4.00, which is
roughly what you would expect from splitting 9 coins with someone who is good at
collecting them, plus the rounds we end early by dying.

### The arithmetic that says what to do next

Splitting the 200 `coin_collector` rounds by whether we killed ourselves:

| | rounds | our score | their score | margin | killed them | mean death step |
|---|---|---|---|---|---|---|
| we survived | 157 | 5.10 +/- 0.43 | 4.26 +/- 0.31 | **+0.83** | 13.4% | |
| we suicided | 43 | 2.51 +/- 0.58 | 6.60 +/- 0.58 | **-4.09** | 0.0% | 122 |

**The whole margin deficit is the suicides.** In the rounds we survive we are
already ahead by +0.83. Suicide costs about 4.9 score in the round it happens,
in both directions at once: we stop collecting at step 122 of 400, and the
opponent gets the rest of the board to itself.

So the first task-3 lever is not a KILLED_OPPONENT reward. It is bomb safety
with another body on the board. It is worth about 1.4 score against the coin
collector on its own, which is more than the gate needs, and until it is fixed a
hunting reward cannot be measured through a 21.5% suicide rate anyway.

The same split against the peaceful agent says the opposite, which is why the
two halves of this gate need different work: 6 suicides in 200 rounds, and we
kill in 69.6% of the rounds we survive. Removing every suicide there moves the
gate number by about a point. That half is genuinely a hunting problem.

### Priced on paper before writing anything

Rather than guess at a fix, every bomb the agent drops in 120 real rounds was
replayed through four candidate survivability checks, then split by whether that
bomb is the one that killed us. `recall` is the share of *fatal* bombs a
candidate would have blocked, `cost` the share of *survived* bombs it would also
have blocked. Blocking bombs is not free; the agent needs them for crates.

vs `coin_collector_agent`, 17 fatal bombs and 5438 survived bombs:

| candidate | blocks fatal | blocks survived |
|---|---|---|
| the live check (opponents are walls) | 0 of 17, 0% | 0.00% |
| opponents may **move** | 4 of 17, **23.5%** | **0.39%** |
| opponents may **bomb** where they stand | 0 of 17, 0% | 0.06% |
| both | 4 of 17, 23.5% | 0.39% |

Two conclusions, ten minutes, no training run. "Opponents may move" is worth
having and nearly free. "Opponents may drop a bomb" is worth nothing and was
dropped. And, the uncomfortable one: **76% of the self-kills are not explained
by the opponent at all.** An escape existed at drop time by every check we can
write, and the agent walked into its own blast anyway.

### A hypothesis, tested and wrong

The obvious explanation for that 76%: channel 3 was identically zero for every
gradient step this network ever took, so the weights reading it have never seen
a nonzero input, and lighting it up injects an untrained perturbation.

Control: play the same games with channel 3 forced back to zero, so the game is
identical and only what the network is shown differs. 200 rounds each.

| | suicide | margin | kill-in-round |
|---|---|---|---|
| vs `coin_collector`, sees opponents | 21.00% | -0.91 | 6.0% |
| vs `coin_collector`, channel 3 zeroed | 22.50% | -0.71 | 10.0% |
| vs `peaceful`, sees opponents | 3.00% | +11.04 | 71.5% |
| vs `peaceful`, channel 3 zeroed | 2.50% | +10.54 | 64.5% |

**No difference anywhere.** Blinding the agent to opponents entirely changes
nothing measurable, which kills the untrained-channel story and also says
something blunter: the current policy is not using channel 3 at all. It is not
that it uses it badly. It plays exactly the same game with the channel on or
off. Any task-3 behaviour has to be trained in; there is nothing to tune.

### The mask, written and measured

`opponent_reach()` is a free-space breadth-first walk outward from every
opponent at once, giving the fewest moves an opponent needs to stand on each
tile. `steps_to_safety` and `escape_exists` take an `opponents_move` flag that
skips a candidate tile when `reach[tile] <= d + 1`.

Off by default in the function signature, and `DQN_OPPONENT_MASK` defaults to on
at play time and off in training. That is deliberate: `train.py` labels
SUICIDAL_BOMB with the same `escape_exists`, and changing that would move the
training signal and the `suicidal_bombs` column underneath every comparison we
have. Same reasoning as the suicide mask.

400 frozen rounds per arm, one variable, in the STATUS table above. Against the
coin collector, self-kills 20.25% -> **13.25%** and margin -0.38 -> **+0.17**,
for 0.17 ms per step and no loss of coins or crates. Against the peaceful agent,
nothing, which is exactly what the pricing predicted.

Note the mask beat its own forecast: drop-time recall was 23.5% of fatal bombs,
which predicted about 5 points of suicide rate, and it delivered 7. Blocking a
bad bomb also avoids the trouble that follows it.

12 new checks in `tools/test_others.py` cover it, including that
`opponent_reach` is 0 on the opponent and unreachable through stone, that a side
pocket the opponents cannot reach first still counts as an escape (the first
version of that test was wrong, not the code), and that with no opponents on the
board the two checks cannot disagree.

### Setting up the first task-3 training run

The reward function needed nothing. `KILLED_OPPONENT` is already 5.0,
`GOT_KILLED` -5.0, and `GOOD_BOMB` already counts opponents caught in the blast
alongside crates. So the first run can be a clean single-variable experiment:
the same settings that produced r3750, plus an opponent in the game. No new
reward, no new event, no new feature. If that alone does not move the numbers we
will know it is the reward and not the exposure.

What did need building is `tools/train_curve.py`, which could only train and
measure solo. It now takes `--opponents`, which puts them in the training game
and in every per-checkpoint evaluation, and writes the task-3 columns.

**A bug caught before it cost an hour.** The first version evaluated every
checkpoint in the training script's own process. `callbacks.MODEL_FILE` is a
module-level constant built at first import, and the agent module is cached in
`sys.modules` afterwards, so all sixteen checkpoints would have been measured
with the weights of the first one and the curve would have come out flat. That
is the same failure as the relative-path bug at the top of this entry, one level
up. Fixed by shelling out to `tools/evaluate3.py --json`, one interpreter per
checkpoint, and verified: a deliberately untrained checkpoint dropped in among
real ones reports 0.00 coins and a -13.40 margin instead of copying its
neighbour.

Cost, measured rather than guessed: 0.75 s per training round in the cloud
container with an opponent on the board, so 4000 rounds is about 50 minutes
there and should be roughly 20 to 25 on the M3 Max, plus about 10 minutes to
evaluate 16 checkpoints at 40 rounds each.

**Before the run:** `dqn-model.pt` still holds the `crate-light` r6000 weights,
and `--resume` reads exactly that file. Copy r3750 in first or the run resumes
from the wrong policy.

```
cp models/task2-classic-r3750-ego13.pt agent_code/dqn_agent/dqn-model.pt

DQN_VIEW="ego:13" DQN_EPS_START=0.3 DQN_EPS_DECAY=2000 \
  python tools/train_curve.py --rounds 4000 --checkpoint-every 250 \
  --eval-rounds 40 --scenario classic --opponents coin_collector_agent --resume
```

`coin_collector_agent` rather than `peaceful_agent` on purpose: it is the harder
opponent, it is the gate that a policy change can actually move, and bomb safety
learned with a real body competing for the same coins should transfer down to
the peaceful matchup. Training against a random walker teaches very little.

Baselines this run has to beat, all frozen, mask on:

| | value |
|---|---|
| vs `coin_collector`: suicide | 13.25% |
| vs `coin_collector`: score margin | +0.17 +/- 0.47 |
| vs `peaceful`: kill-in-round | 69.2% |
| solo `classic`: coins (must not collapse) | 7.51 |

That last row matters. Nothing in this run protects task 2, so the winning
checkpoint has to be re-measured solo as well before it can ship.

### Reproducibility: the peaceful matchup is not seedable

`peaceful_agent` calls `np.random.seed()` with no argument in its own `setup`,
which reseeds numpy from OS entropy and overrides the world's `--seed`. Two
200-round runs of the same frozen weights gave **74.5%** and **68.5%**
kill-in-round. Both sit inside each other's confidence interval, so this is not
a bug, but it does mean the 80% line cannot be called at n=200. Re-measure at
1000 rounds before claiming that half of the gate either way.

### New tooling

`tools/evaluate3.py`. The framework's `round_statistics` sums `coins`, `kills`
and `suicides` **over all agents**, which was fine solo and is useless now:
`coin_collector_agent` does drop bombs, so an aggregated kill count cannot say
who killed whom. `evaluate3.py` drives `BombeRLeWorld` directly and reads each
agent's `statistics` dict at the end of every round, before `start_round`
clears it. Same world class, same `do_step`, same stopping rules as `main.py`;
only the bookkeeping is different. It reports a per-round score margin with a
confidence interval, kill-in-round as a Wilson interval, and `--gate 3`.

### What is still unknown

- **Both gates now need training, and there are 7 days.** The coin-collector
  margin is +0.17 +/- 0.47, a tie; the peaceful kill rate is 69.2% against 80%.
  Neither closes without a policy that behaves differently when an opponent is
  on the board, and the blinding control says the current one does not.
- Whether the peaceful kill rate is limited by hunting or by opportunity. 69%
  of rounds already contain a kill with **zero** kill-directed behaviour, so
  most of those kills are the peaceful agent random-walking into a crate blast.
  Worth knowing before paying for a reward that chases something already
  happening by accident.
- What the remaining 13.25% of self-kills are. They are not the opponent
  standing in the way (the mask handles that now) and not the opponent's bombs
  (priced at zero). Most likely the escape is real at drop time and the policy
  fails to walk it, which is a training problem, not a masking one.
- The augmentation ablation is dropped for now. It buys sample efficiency for a
  task-2 gate we already decided to advance past, and task 3 has two gates that
  need the remaining days.


## 2026-09-09 (night, 7) — Daniela's symmetry suggestion, implemented

The board is square and every rule of the game is invariant under the eight
symmetries of a square, so one observed transition is really eight. Rotating a
board 90 degrees does not change which move is correct, it changes what that
move is *called*, and until now the network had to learn each orientation from
separate experience.

### Design choices

**Applied at sampling time, not at push time.** Pushing eight copies of every
transition would cut the number of *distinct* situations a 50,000-entry buffer
holds by a factor of eight. Transforming the batch as it is drawn keeps the
buffer's diversity, costs no memory, and replays each transition in a different
orientation on every visit, which also regularises.

**Per sample, not per batch.** One orientation for a whole batch would correlate
every gradient in it. Implemented by grouping the batch by transform, so it is
eight small tensor ops rather than a Python loop.

**Training only.** The submitted agent never samples a replay buffer, so this
cannot affect the submission. `DQN_AUGMENT` defaults to 0 so the runs already in
`experiments/runs/` stay the honest baseline.

### The part that needed testing

`state_to_features` builds channels as `channels[c][x, y]`, so tensor dim -2 is
x. Under rot90 in that plane an offset (u, v) maps to (-v, u), which sends
UP -> RIGHT -> DOWN -> LEFT, one step along `ACTIONS[0:4]`. Flipping x fixes UP
and DOWN and swaps RIGHT and LEFT. WAIT and BOMB are invariant.

A wrong action permutation here would teach the network that walking left leads
where walking up leads, and **nothing in the loss curve or the reward would look
wrong**. Given how much of this project has been lost to silent errors of
exactly that shape, it gets a real test suite: `tools/test_augment.py`, 71
checks, all passing.

The strongest of them is check 3: transform the *world* (rotate the arena, move
the agent and the coins accordingly), compute features from that, and compare
against transforming the features of the original. They agree exactly for all
eight elements. That validates the board transform independently of the algebra.
Check 2 then pins the action permutation by placing a coin one tile in each
direction and confirming it lands where the permuted action points.

`next_legal` moves by the *inverse* permutation, since it is a mask indexed by
action rather than an action.

Cost: 2.51 ms per sampled batch of 128 against 0.53 ms, so about 70 seconds
added across a 4000-round run. Irrelevant.

### The ablation, set up as a single-variable comparison

`dqn-model.pt` has been reset to `archive-crate-light-6000r/dqn-r006000.pt`, the
exact weights the baseline `classic` run resumed from. The augmented run
therefore differs from `2026-09-09-task2-classic-4000r` in one flag and nothing
else, which is the standard the plan sets for every comparison.

Baseline to beat: **7.51 coins over 1000 rounds, 0.60% suicide**, from r3750.


## 2026-09-09 (night, 6) — Submission dry run passes, 8 days early

The 17.09 submission test is a hard deadline and the plan lists it as a common
way teams lose the tournament, so it is now de-risked rather than assumed.

Ran the agent exactly as the tutors will: **stock `settings.py`** (our
`crate-light` scenario stripped, since only `agent_code/dqn_agent/` is
submitted), **no `DQN_*` environment variables at all**, `train=False`, against
three `random_agent`s on `classic`, weights loaded from
`agent_code/dqn_agent/dqn-model.pt`.

```
rounds 20 | score 158 | invalid actions 0 | 0.73 ms/step | 7.9 coins/round
exit code 0
```

Everything that was flagged as a submission risk holds:

- `peek_view` reads the view from the checkpoint before building the network, so
  `DQN_VIEW` being unset does not build the wrong shape. This was open risk #1
  after task 1.
- `DQN_LEGAL_MASK` and `DQN_SUICIDE_MASK` both default to on at play time, so
  the unset environment gives the intended behaviour rather than the ablation.
- 0.73 ms per step against the 500 ms tournament budget, single thread.
- No dependency on our added scenario.

Only 20 rounds; this is a crash-and-configuration check, not a performance
measurement.


## 2026-09-09 (night, 5) — `classic`: 7.76 coins, and a curve that lied again

Resumed the converged `crate-light` policy onto `classic` (119 crates against
56), 4000 rounds, `DQN_EPS_START=0.3`, `DQN_EPS_DECAY=2000`.

### First: the curve was wrong, again

`eval_curve.csv` came out with 24 rows going to r6000 for a 4000-round run, and
the last eight rows averaged about 4.3 coins, which read as a failure.

`train_curve.py` builds the curve with `sorted(CHECKPOINTS.glob("dqn-r*.pt"))`.
The 4000-round run overwrote r250 through r4000, but r4250 through r6000 from
the previous 6000-round `crate-light` run were still sitting in the directory.
So the tail of the curve was **a different run's models, from a different
scenario, measured on this one**. Rows 1-16 are the real run.

This is the fifth time a measurement artefact has looked like a modelling
result. Fixed two ways: `archive_checkpoints()` moves any existing checkpoints
into a timestamped subfolder before a training run starts, and the curve now
also refuses to plot any checkpoint numbered above `--rounds`. The archive step
additionally prevents one run silently overwriting another's weights, which
cost us run 1's checkpoints earlier the same day.

### The training run

```
rounds       eps   steps  coins  crates   good  suicid   reward    loss
1-500      0.269    32.6   0.32   12.16   3.48   0.034    +0.39  0.2597
1001-1500  0.144    59.0   0.94   25.98   6.78   0.008   +13.71  0.1494
1501-2000  0.081   122.2   2.43   50.12  13.35   0.014   +39.52  0.0887
2001-2500  0.050   143.5   3.04   59.36  15.88   0.008   +50.20  0.0554
2501-3000  0.050   150.0   3.31   62.22  16.76   0.028   +53.62  0.0560
3501-4000  0.050   145.1   3.23   61.98  16.53   0.018   +52.88  0.0566
```

Converged by round 2500. 62 crates destroyed per round against 34 on
`crate-light`, 16.5 good bombs, invalid actions 0 across all 4000 rounds.
The resume at `DQN_EPS_START=0.3` worked: the agent starts at 12 crates per
round rather than the ~1.9 a from-scratch run starts at.

### Confirmed over 300 frozen rounds each, on `classic`

| checkpoint | coins | 95% CI | steps | suicide | 95% upper | reach 8+ |
|---|---|---|---|---|---|---|
| r2250 | 6.28 | [5.99, 6.57] | 395.5 | 0.67% | 2.40% | 39.0% |
| r2750 | 7.28 | [7.02, 7.55] | 392.9 | 1.33% | 3.38% | 65.7% |
| r3000 | 7.74 | [7.52, 7.96] | 372.5 | 3.00% | 5.60% | 72.7% |
| **r3750** | **7.76** | [7.53, 7.98] | 395.8 | **0.67%** | 2.40% | **75.0%** |
| r4000 | 6.34 | [6.09, 6.60] | 398.7 | **0.33%** | **1.86%** | 35.3% |
| `rule_based_agent` | **8.60** | +/- 0.09 | | 0.00% | | **93.0%** |

**The agent is better on the harder board.** 7.76 on `classic` against
7.02-7.48 on `crate-light`. More crates means more coins per bomb and the coin
count is 9 either way, so the "harder" board is actually the easier one for this
objective. Worth stating in the report, since the curriculum ordering assumed
the opposite.

### The 1000-round re-confirmation of the crate-light claim

The 0.00% suicide rate claimed for `crate-light` r5750 was partly luck, as
suspected. Over 1000 rounds it is **1.20%, 95% upper bound 2.09%**, with 6.91
coins. So that checkpoint does *not* cleanly pass the 2% gate; it sits on the
boundary. r5000 over 1000 rounds: 7.30 coins, 4.90% suicide. Recording this
because the 300-round number is already written down above and should not be
the one that survives into the report.

### Confirmed over 1000 rounds: the suicide gate is met

`classic` r3750, 1000 frozen rounds: **7.51 coins [7.37, 7.65], suicide 0.60%
with a 95% upper bound of 1.30%**, 394.5 steps, 70.6% of rounds reach 8+.

**Gate 2 survival: PASSED.** The whole confidence interval is below the 2%
limit, which is the standard used for gate 1 and the only one that supports a
claim in the report. **Gate 2 coins: 7.51 of 9 against a target of 8, short by
0.49.**

### What is left

0.49 coins. `classic` has converged, so the next lever is not more rounds: it
is the symmetry augmentation, which buys sample efficiency, which is exactly
what a converged-but-short agent needs.


## 2026-09-09 (night, 4) — 6000 rounds: the suicide half of gate 2 passes

Same command at double the budget, `DQN_EPS_DECAY=3000`, mask off in training
and on at play. `suicidal_bombs` totals 582 over the run, confirming the mask
really was off; invalid actions 0.

### It converged

```
rounds       eps   steps  coins  crates   good useless  suicid   reward    loss
2501-3000  0.129    56.7   0.36    7.12   3.65    2.99   0.168    -1.30  0.3774
3001-3500  0.050   153.2   1.35   19.53   9.53    5.68   0.050   +16.79  0.1431
3501-4000  0.050   165.8   3.17   31.64  13.91    3.85   0.070   +33.82  0.0669
4001-4500  0.050   165.8   4.01   33.81  14.65    6.11   0.050   +38.46  0.0606
4501-5000  0.050   159.8   4.07   33.57  14.36    3.33   0.042   +37.49  0.0560
5001-5500  0.050   159.7   4.06   34.31  14.83    3.90   0.040   +38.79  0.0569
5501-6000  0.050   158.0   4.07   33.72  14.82    4.23   0.042   +38.46  0.0566
```

Flat on every metric for the last 2000 rounds. This is the first task-2 run that
plateaued rather than being cut off mid-climb, so "train longer on
`crate-light`" is now exhausted as a lever.

`eval_curve.csv` oscillates between 5.78 and 7.72 after round 4000, which is
40-round noise (+/- 0.85), not signal. Confirmed the candidates over 300 rounds:

| checkpoint | coins | 95% CI | steps | suicide | reach 8+ |
|---|---|---|---|---|---|
| r4250 | 6.32 | [6.03, 6.61] | 386.7 | 3.67% | 42.7% |
| **r5000** | **7.48** | [7.20, 7.76] | 346.0 | 4.67% | **71.7%** |
| r5250 | 6.63 | [6.33, 6.94] | 382.4 | 4.00% | 52.0% |
| **r5750** | 7.02 | [6.78, 7.27] | **399.0** | **0.00%** | 54.3% |
| r6000 | 5.56 | [5.21, 5.91] | 382.2 | 1.00% | 35.3% |
| run 1 r3000 (previous best) | 6.92 | [6.64, 7.19] | 371.5 | 7.33% | 55.7% |
| `rule_based_agent` | **8.99** | +/- 0.01 | 340.7 | 0.00% | **100.0%** |

**Gate 2, suicide rate below 2%: passed** by `r5750` (0 of 300) and `r6000`
(1.00%). **Gate 2, 8 of 9 coins: not passed**, best is 7.48.

### What the numbers say the problem is now

`r5750` survives 399 of 400 steps and still finds only 7 coins. It is not dying,
it is running out of time. `rule_based_agent` finds 8.99 on the same board in
341 steps, so the target is reachable and our agent is roughly one coin and
sixty steps of efficiency short of it.

There is also a visible safety/greed trade-off along one run: `r5000` finds the
most coins and reaches 8+ in 71.7% of rounds while dying 4.67% of the time;
`r5750`, 750 rounds later, never dies and finds half a coin fewer. Both are kept.

### Honest caveat on the pass

The safety metric swings hard between adjacent checkpoints (4.67%, then 0.00%,
then 1.00%, at 250-round spacing). The candidates were selected on *coins* from
the 40-round curve and the suicide rate was then measured fresh, so it is not
circular, but 0 of 300 from a checkpoint whose neighbours sit at 1-5% is partly
luck. Re-confirming over 1000 rounds before putting "passes gate 2 on survival"
in the report.

### Still not done

Daniela's symmetry suggestion (8-fold dihedral augmentation) is still not
implemented. It was deferred while the reward function was broken, which was the
right call, but that reason has now expired: the run converges, the bugs are
gone, and the remaining problem is sample efficiency, which is exactly what
augmentation buys.


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
