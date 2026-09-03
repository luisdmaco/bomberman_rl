# Team kickoff: talking track

Roughly 15 minutes. Send the roadmap link the day before so nobody is reading during the meeting.

Your job here is not to teach reinforcement learning. Your teammates sat in the same lectures. Your job is to get four decisions agreed and the week started.

---

## 1. What we're actually building (2 min)

> "Four agents on a grid, 400 steps. You move, drop bombs, collect coins. Coins are 1 point, killing someone is 5. Our agent has half a second per move.
>
> The thing that kills you is your own bomb. That's not a joke, it's where most teams lose."

## 2. How the agent learns, in plain words (4 min)

Don't reach for equations. This chain works:

> "Every step the agent picks one of six moves. We give each move a score: how many points do I expect if I do this. Then we just take the highest one. That's the whole decision rule.
>
> The scores start as garbage. Random numbers. Then it plays. It steps into a blast and dies, so we lower that score. It walks onto a coin, so we raise it. Do that a few hundred thousand times and the numbers stop being garbage.
>
> Here's the catch, and it's the hard part of the project. It doesn't die because of the last move. It dies because it dropped a bomb four steps earlier and then walked into its own blast. So the blame has to travel backwards from the death to the actual mistake, and that's slow.
>
> We speed it up two ways. One, punish the bad decision the moment it happens instead of waiting: drop a bomb with no escape route, instant penalty. Two, make sure the agent can even see the difference. If our features describe 'bomb here with an escape' and 'bomb here with no escape' using the same numbers, it can never learn to tell them apart, no matter how long we train."

If someone asks what a DQN is:

> "Same idea, but a neural network computes the six scores instead of looking them up in a table. We're doing the table version first because it trains in minutes and we can actually read it when it misbehaves."

## 3. The roadmap, and why it's shaped like this (3 min)

> "Three weeks. Code is due 21.09, report 28.09. There's a free crash test on 17.09 where the tutors run our agent and send back the errors, so that's really our deadline.
>
> Four stages, each with a number we have to hit before moving on. Collect coins. Bomb crates without dying. Hunt the weak agents. Beat the rule-based agent.
>
> Stage two is the whole project. Everything after it is easier. The plan puts it in week 1 on purpose, which leaves eleven days of slack. That slack is insurance, not free time."

Then the point that matters most:

> "The tournament is worth points, but the spec says the science carries much more weight, and marks the results section as the most important part of the report, in bold. So the rule is: every change we make gets measured before and after. If we can't plot it, we didn't do it. That's why the first thing we build is the measuring script, not the agent."

## 4. Decisions we need today (4 min)

1. **Team name.** Needed for registration, and it's also our folder name and our tournament identity. Decide it in this meeting, not later.
2. **Who owns what.** Split by component, not by model: features and game logic / learning algorithm / evaluation and experiments. Rotate onto stage two together when we get there.
3. **Agree on the measuring discipline.** No undocumented changes. Every run saved with its config and result.
4. **Agree the DQN is allowed to fail.** If it isn't clearly better by 20.09, the simple agent ships and the DQN goes in the report as an approach we explored. Nobody gets to be sad about this in week 3.

## 5. This week (2 min)

Register the team. Public repo, everyone on it. Build the evaluation script and get baseline numbers for the four provided agents. Write the feature extraction. Have something collecting coins by Sunday.

---

## Pushback you should expect

**"Why not go straight to deep learning?"**
> "Because the spec warns twice that teams have shown up at the deadline with an unconverged network. And because the grade is the experimental record, not the ceiling of the method. We can still build one, it just isn't what we bet the submission on."

**"Isn't a lookup table too simple to win?"**
> "Past winners used exactly this. The spec says so directly. A simple model with a rigorous experiment log scores better than a fancy one without."

**"Can we each build our own agent and submit the best?"**
> "No, and this one isn't negotiable. The spec explicitly forbids splitting labour so each person owns a separate model, and says they attach great importance to real teamwork. We split by component."
