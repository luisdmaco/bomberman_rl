# Bomberman RL: Project Plan

Machine Learning Essentials, Summer Semester 2026. Team of 3.

| Milestone | Deadline | Notes |
|---|---|---|
| Team registered (name + members) | ASAP, week of 31.08 | tinyurl.com/fml-final-project-teams, plus MaMPF display names and team invitation code |
| Submission test upload | Thu 17.09.2026, 21:00 | Tutors run your agent with `train=False` vs three random agents and send back stack traces |
| Agent code (final) | Mon 21.09.2026, 21:00 | `final-project-agent-code.zip`, one `agent_code/` subdirectory only |
| Report | Mon 28.09.2026, 21:00 | PDF, ~12000 words for 3 members, every section labelled with its author |

Public repo with the full codebase is required and its URL must appear in the report. The report itself must **not** go into that repo.

---

## 1. What actually decides the grade

Read section 4 and 9 of the spec carefully. Tournament placement is a factor, but the dominant factor is:

> "a systematic (scientific) approach to agent design, optimization, and testing will carry much more weight"

and section 6 of the report structure is called out in bold as "the most important section". So the plan below is built around one principle: **every design change is an experiment with a recorded before/after number.** If we cannot plot it, we did not do it.

Practical consequence: the evaluation harness is built in week 0, before the first learning agent exists. Every checkpoint we ever train stays on disk with its config, so on 22.09 we can regenerate every figure without retraining anything.

---

## 2. Models

The spec requires at least two different models, at least one using techniques from the lecture, and warns explicitly that deep learning teams have missed the deadline with unconverged networks.

**Model A: Q-learning with hand-crafted features (primary, tournament candidate).**
Compact engineered feature vector, tabular Q-table over the discretised feature space first, then linear function approximation / regression forest on the same features if the table proves too coarse. Trains in minutes on a laptop CPU, is fully inspectable (you can print the Q-table row for a state and see why it acted), and matches what past winners of this competition used. This is our safe submission.

**Model B: Deep Q-Network (comparison, upside).**
Small CNN over stacked board channels (field, self, others, bombs, danger, coins), Double DQN with a target network and prioritised replay. Learns its own features, so the report gets a genuine "engineered features vs learned features" comparison. GPU allowed for training, but inference must run on CPU inside 0.5 s/step.

**Rule:** Model B never blocks Model A. If on 20.09 Model B is not clearly better on our head to head metric, Model A ships and Model B is written up honestly as the approach we explored, with its training curves. That is a perfectly good report result, not a failure.

The spec forbids splitting labour so that each member owns a separate model. We split by **system component** instead (see section 7), and both models are built by the whole team on a shared feature and evaluation stack.

---

## 3. Curriculum: four tasks, four gates

Each gate has a numeric pass criterion. We do not move to the next task until the gate is met, and each gate is a row in the results table of the report.

**Task 1: navigation, no crates, no bombs.**
`python main.py play --no-gui --agents our_agent --train 1 --scenario coin-heaven`
Gate: collects >= 45 of 50 coins within 400 steps, averaged over 200 rounds, with zero invalid actions.

**Task 2: crates and bomb survival. This is the one that matters most.**
`--scenario classic`, solo. The agent must place bombs to open crates and never blow itself up.
Gate: suicide rate < 2% over 200 rounds, and >= 8 of 9 coins collected on average.
The spec says outright: "Escaping bombs is a crucial capability for good tournament performance, so place proper emphasis on this step." Most tournament losses are self-inflicted. Budget the most time here.

**Task 3: hunting.**
vs `peaceful_agent` (easy) and `coin_collector_agent` (hard).
Gate: kills the peaceful agent in >= 80% of rounds; positive score margin vs coin_collector.

**Task 4: full game.**
vs `rule_based_agent`, and self-play against earlier versions of ourselves.
Gate: mean score over 200 rounds strictly above `rule_based_agent`'s. The spec is blunt that beating the rule based agent is the entry ticket for the tournament.

---

## 4. Feature design (Model A)

Design constraint: small enough that a Q-table over it converges in a few hundred thousand steps, rich enough that the optimal action is a function of it. Target roughly 10 features with 2 to 5 values each.

Candidate v1 set:

1. **Coin direction.** BFS over free tiles from our position to the nearest reachable coin; value in {up, right, down, left, none}.
2. **Crate/target direction.** Same BFS to the nearest tile from which a bomb would destroy at least one crate.
3. **Opponent direction.** BFS to the nearest opponent, bucketed by distance (adjacent / near / far / none).
4. **Neighbour passability.** For each of the 4 neighbours: free / crate / wall / occupied. 
5. **Neighbour danger.** For each of the 4 neighbours and the current tile: steps until that tile becomes lethal (0 = lethal now, large = safe). Computed from `bombs` plus `explosion_map`.
6. **Bomb available.** Boolean from `game_state['self'][2]`.
7. **Escape exists if I bomb now.** Boolean. Simulate dropping a bomb here, build the resulting danger timeline, BFS for a tile reachable strictly before it becomes lethal. This single feature is the difference between an agent that kills itself and one that does not.
8. **Escape direction when in danger.** Direction of the first step on the shortest path to a safe tile, or none.
9. **Crates destroyed if I bomb here.** Bucketed 0 / 1 / 2 / 3+.
10. **Dead end indicator.** Whether the current tile has exactly one free neighbour.

We start with 1, 4, 5, 6, 7, 8 for task 1 and 2, and add the rest as the tasks demand them. Each added feature is an ablation row in the report.

**Symmetry augmentation.** The board has the 8-fold dihedral symmetry (4 rotations x mirror). Every transition can be replayed under all 8 transforms with the action relabelled, giving 8x the data for free and removing directional bias. Design the feature encoding so a transform of the state is a cheap permutation of the feature vector. This is called out in the spec as a hint and is a strong, cheap win, so build it into the feature module from day one rather than bolting it on later.

---

## 5. Reward shaping

Base rewards are sparse (1 per coin, 5 per kill), so we need a dense auxiliary signal. Spec section 7 gives the design rules, and we follow them literally:

- **Pair every positive with a negative of equal or greater magnitude.** Reward for stepping towards the nearest coin means an equal penalty for stepping away or waiting, otherwise the agent oscillates to farm the positive.
- **Prefer potential-based shaping.** Ng, Harada and Russell (1999), cited in the spec footnote: `F(s, s') = gamma * Phi(s') - Phi(s)` with `Phi(s) = -d_BFS(s, nearest coin)` leaves the optimal policy unchanged. Use this for the navigation terms so we cannot accidentally teach a wrong optimum.
- **Many small custom events are cheap.** Adding events costs almost nothing computationally compared to adding features.

Custom events to define in `train.py`:
`MOVED_TOWARDS_COIN`, `MOVED_AWAY_FROM_COIN`, `ENTERED_DANGER`, `LEFT_DANGER`, `WAITED_IN_DANGER`, `USELESS_BOMB` (no crate, no opponent in blast), `GOOD_BOMB` (>=1 crate or opponent in blast with a valid escape), `SUICIDAL_BOMB` (no escape exists), `OSCILLATION` (returned to a position visited within the last 4 steps), `SURVIVED_STEP`.

Terminal: heavy negative on `KILLED_SELF` and `GOT_KILLED`, heavy positive on `KILLED_OPPONENT`.

**Overfitting warning from the spec:** auxiliary rewards do not exist in official games. Before every submission, evaluate on raw game score only, never on shaped return.

---

## 6. Evaluation harness (built first, week 0)

A single script, `tools/evaluate.py`, that runs N rounds headless with a fixed seed list and emits a JSON row per round. `main.py` already supports `--n-rounds`, `--seed`, `--no-gui`, `--match-name` and `--save-stats` (which dumps results to JSON), so the harness wraps those rather than reimplementing them. Metrics:

| Metric | Why |
|---|---|
| Mean score per round (+ 95% CI) | The tournament criterion |
| Coins collected | Task 1 and 2 progress |
| Opponents killed | Task 3 and 4 progress |
| Suicide rate | The single biggest failure mode |
| Survival rate / steps survived | Correlates with score in the tournament |
| Invalid action rate | Pure bug detector, should go to 0 |
| Mean decision time per step | Must stay far under 0.5 s |

Protocol for every comparison: N >= 200 rounds, identical seed list across variants, report mean with a 95% confidence interval, change one thing at a time. Training curves (running mean reward, TD error, epsilon schedule) get logged to CSV per run and plotted with a shared plotting script so every figure in the report has the same style.

Store every run under `experiments/<date>-<name>/` with the config, checkpoint, CSV log and result JSON. The report is then assembled from files, not from memory.

---

## 7. Team split (3 members)

Split by component, not by model, and pair on the hard parts.

- **Member 1: environment interface.** `state_to_features`, BFS utilities, danger map, bomb simulation, symmetry transforms, unit tests for all of it. This code is used by both models.
- **Member 2: learning.** Q-learning update, replay buffer, epsilon schedule, checkpointing, the DQN in week 1 and 2. Reward shaping is co-designed with member 1.
- **Member 3: evaluation and experiments.** Harness, metrics, plots, hyperparameter sweeps, the experiment log, Docker submission test. Owns the results section skeleton from day one.

Rotate: whoever finishes a component early pairs on task 2 bomb escape, which is the critical path. Everyone reviews everyone's PRs. Every member writes and signs sections of the report covering work they actually did, since sections must be attributed for legal reasons.

---

## 8. Week by week

**Week 0, 31.08 to 06.09: foundation.**
Register team and name. Repo hygiene (public repo, `.gitignore` for checkpoints and logs, requirements pinned). Evaluation harness running against the provided agents to produce baseline numbers for `random_agent`, `peaceful_agent`, `coin_collector_agent`, `rule_based_agent`. Feature module v1 with unit tests. Tabular Q-learning agent passing **Task 1 gate** by Sunday.

**Week 1, 07.09 to 13.09: bombs.**
Bomb simulation and escape feature. Reward shaping v1. **Task 2 gate** (suicide rate < 2%) by Wednesday, this is the critical path and gets the whole team if it slips. Then Task 3 vs peaceful and coin_collector. DQN skeleton starts in parallel once Task 2 is passed. First ablation table.

**Week 2, 14.09 to 20.09: compete and harden.**
Task 4 vs `rule_based_agent`. Self-play against frozen earlier checkpoints. Hyperparameter sweep (learning rate, gamma, epsilon schedule, replay size, reward magnitudes). **Wed 16.09: full Docker test locally, stock `settings.py`, `train=False`.** **Thu 17.09 21:00: upload for the official submission test.** Fix whatever comes back over the weekend. **Sun 20.09: training freeze.**

**Week 3, 21.09 to 28.09: submit and write.**
Mon 21.09: final zip uploaded well before 21:00. Then the report, which should mostly be assembly since the experiment log and figures already exist. Draft by Thu 24.09, internal review Fri, polish over the weekend, submit Mon 28.09.

Buffer note: the plan front-loads. Task 2 done by 09.09 leaves 11 days of slack before the code deadline. Treat that slack as insurance, not as spare time.

---

## 9. Framework details that cost teams the tournament

Verified against `environment.py`, `items.py` and `settings.py` in this repo:

1. **Explosions pass through crates.** `Bomb.get_blast_coords` only breaks on stone walls (`arena == -1`). A danger map that assumes crates block the blast will get the agent killed. Blast reach is `BOMB_POWER = 3` tiles in each of the four directions.
2. **Step order inside `do_step`:** agents act, then coins collected, then explosions age, then bombs tick and detonate, then kills evaluated. So a bomb showing `timer == 0` in the state passed to `act` detonates at the end of *this* step: moving off its blast this step still saves you. Coins are collected *before* kills are evaluated, so you can bank a coin and die in the same step.
3. **You get exactly 4 actions after dropping a bomb** before it explodes (`BOMB_TIMER = 4`), which is just enough to clear a 3-tile blast and turn a corner. Any escape check must use this exact budget.
4. **`explosion_map`** gives remaining dangerous steps per tile; any nonzero value is lethal. `EXPLOSION_TIMER = 2`, so a fresh blast is dangerous for two steps, then becomes harmless smoke.
5. **Only the agent subdirectory is submitted.** Any change to `settings.py`, `environment.py` or anything else outside `agent_code/<our_agent>/` is absent in the tournament. Test with stock settings before every submission.
6. **Your agent code runs with the working directory set to its own folder.** `SequentialAgentBackend.send_event` in `agents.py` wraps every callback in an `os.chdir` into `agent_code/<name>/` and restores it afterwards, which is why the template's bare `"my-saved-model.pt"` lands in the agent directory rather than the repo root. The commented-out `ProcessAgentBackend` does not do this. Use `Path(__file__).parent / "model.pt"` anyway: it is correct under both backends, and absolute paths are the single most common submission crash.
7. **The `self.train == False` code path must work standalone.** The official test runs one game with `train=False` against three random agents, and `train.py` is not even imported then.
8. **No multiprocessing in the final agent.** Fine during training. Tournament hardware: one thread of a Ryzen 5 2600, 8 GB RAM, CPU only, 0.5 s per step, and an overrun is subtracted from the next step's budget.
9. **Log `numpy`, `pygame`, `tqdm` and anything else in `requirements.txt`** and state the extra libraries at the start of the report.

---

## 10. Report skeleton (start it in week 0, not week 3)

Structure fixed by the spec, ~12000 words total, each section headed with its author.

1. Introduction: the problem and why it is hard.
2. Background: RL approaches considered (Q-learning, SARSA, n-step, function approximation, DQN variants), with references.
3. Project planning: this document, how it survived contact with reality, hardware for DQN training.
4. Methods: features, reward design, both models, and the testing methodology with defined metrics.
5. Training: the curriculum, self-play, replay prioritisation, every speed-up trick used.
6. **Experiments and results: the heaviest section.** Training curves, ablations, head to head tables with confidence intervals, and an honest account of what failed and why.
7. Conclusion: findings, what we would do with more time, feedback on the game setup.

Keep a running `experiments/LOG.md` from day one, one dated entry per experiment with hypothesis, change, result. Section 6 then writes itself.

---

## 11. Immediate next actions

1. Agree the team name and register at tinyurl.com/fml-final-project-teams; set MaMPF display names; join the team submission with the invitation code.
2. Push this repo to a public GitHub repo, all three members with access.
3. Create `agent_code/<team_name>_agent/` from `tpl_agent`.
4. Build `tools/evaluate.py` and record baseline numbers for the four provided agents. These baselines are table 1 of the report.
5. Write `state_to_features` v1 with unit tests, plus BFS and danger map utilities.
6. Train the first tabular Q agent on `coin-heaven` and hit the Task 1 gate.
