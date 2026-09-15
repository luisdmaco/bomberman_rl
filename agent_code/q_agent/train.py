from .features import (state_to_features, danger_map, get_action_mask,
                       bfs_direction_and_distance, get_blast_coords,
                       bfs_to_opponents, escape_exists, opponent_in_blast)
import events as e
import csv
from collections import deque
import copy

# one row per round next to model.npy, for the training curves
LOG_COLUMNS = ["round", "steps", "reward", "td_abs_mean", "epsilon", "lr",
               "coins", "crates", "bombs", "kills", "suicides", "got_killed",
               "invalid", "bombed_opponent", "trapped_opponent",
               "opp_max_score", "win"]

IN_DANGER = "IN_DANGER"
MOVED_CLOSER_TO_COIN = "MOVED_CLOSER_TO_COIN"
MOVED_FARTHER_FROM_COIN = "MOVED_FARTHER_FROM_COIN"
BOMBED_CRATES = "BOMBED_CRATES"
ESCAPED_DANGER = "ESCAPED_DANGER"
STALLED = "STALLED"
MOVED_CLOSER_TO_OPPONENT = "MOVED_CLOSER_TO_OPPONENT"
MOVED_FARTHER_FROM_OPPONENT = "MOVED_FARTHER_FROM_OPPONENT"
BOMBED_OPPONENT = "BOMBED_OPPONENT"
TRAPPED_OPPONENT = "TRAPPED_OPPONENT"
# FOURTH FIX
WAITED_WITH_OPPONENTS = "WAITED_WITH_OPPONENTS"
# P5
AMBUSH_READY = "AMBUSH_READY"
WIN_ROUND = "WIN_ROUND"

# TODO: are these the best values for our case?
# Hyperparameter training
def setup_training(self):
    
    self.epsilon = 0.2
    self.epsilon_min = 0.05
    self.epsilon_decay = 0.9995 # multiply after every round
    self.lr = 0.005 # SECOND FIX
    self.lr_min = 0.0005
    self.lr_decay = 0.999
    self.gamma = 0.95

    # THIRD FIX:
    # target network: bootstrap off a slow-moving snapshot rather than the
    # weights we're actively updating, so an action whose next state looks
    # like its current state (WAIT, once opponents make it legal everywhere)
    # can't feed its own overestimation straight back into itself every step
    self.target_model = copy.deepcopy(self.model)
    self.target_sync_every = 100

    self.recent_positions = deque(maxlen=10)

    self.log_path = "training_log.csv"
    with open(self.log_path, "w", newline="") as fh:
        csv.writer(fh).writerow(LOG_COLUMNS)
    reset_round_log(self)


def reset_round_log(self):
    self.round_reward = 0.0
    self.round_td_abs = 0.0
    self.round_updates = 0
    self.round_events = {}


def note_events(self, events, reward):
    self.round_reward += reward
    for ev in events:
        self.round_events[ev] = self.round_events.get(ev, 0) + 1

# TODO: does this function updates the actual state of the game
# or keeps track of the successes until the current step
def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    old_f = state_to_features(old_game_state)
    new_f = state_to_features(new_game_state)

    # custom event: agent stays somewhere a blast will reach
    if new_game_state is not None:
        _, _, _, (nx, ny) = new_game_state['self']
        dmap = danger_map(new_game_state['field'], new_game_state['bombs'], new_game_state['explosion_map'])
        if (nx, ny) in dmap:
            events.append(IN_DANGER)

    # 4. FIX:
    if self_action == 'WAIT' and old_game_state is not None and old_game_state['others']:
        events.append(WAITED_WITH_OPPONENTS)
        
    # dense scape signal: reward for leaving a blast zone
    if old_game_state is not None and new_game_state is not None:
        _,_,_, o_pos = old_game_state['self']
        _,_,_, n_pos = new_game_state['self']
        odm = danger_map(old_game_state['field'], old_game_state['bombs'], old_game_state['explosion_map'])
        ndm = danger_map(new_game_state['field'], new_game_state['bombs'], new_game_state['explosion_map'])
        if o_pos in odm and n_pos not in ndm:
            events.append(ESCAPED_DANGER)

    if new_game_state is not None:
        self.recent_positions.append(n_pos)
        if(len(self.recent_positions) == self.recent_positions.maxlen
            and len(set(self.recent_positions)) <= 2):
            events.append(STALLED)
    

    # custom event: immediate reward for a productive bomb
    if self_action == 'BOMB' and old_game_state is not None:
        _, _, _, (ox, oy) = old_game_state['self']
        f = old_game_state['field']
        blast = get_blast_coords(ox, oy, f)
        n_crates = sum(1 for (bx, by) in blast if f[bx, by] == 1)
        for _ in range(n_crates):
            events.append(BOMBED_CRATES) # once per crate -> reward scales with payoff
        # same trick for opponents
        caught = [o for o in old_game_state['others'] if o[3] in blast]
        if caught:
            events.append(BOMBED_OPPONENT)
            # a bomb they can walk out of is worth little, one they cannot is nearly a kill
            hypothetical = list(old_game_state['bombs']) + [((ox, oy), 4)]
            for o in caught:
                rivals = [q for q in old_game_state['others'] if q is not o]
                if not escape_exists(o[3], f, hypothetical, rivals,
                                     old_game_state['explosion_map'],
                                     assume_own_bomb=False):
                    events.append(TRAPPED_OPPONENT)
                    break

    # P5: am I in position to land a bomb on a rival *right now*?
    # (bomb ready, a rival in the blast of a bomb at our feet, and we can
    # escape it). This is the "directed bombing" signal task 3 is missing.
    if new_game_state is not None:
        _, _, new_can_bomb, (nx, ny) = new_game_state['self']
        if (new_can_bomb and new_game_state['others'] and opponent_in_blast(nx, ny, new_game_state['field'],new_game_state['others'])
            and escape_exists((nx, ny), new_game_state['field'], new_game_state['bombs'], new_game_state['others'], new_game_state['explosion_map'])):
            events.append(AMBUSH_READY)


    # dense shaping: avoid back and forth 
    if old_game_state is not None and new_game_state is not None:
        _, _, _, old_pos = old_game_state['self']
        _, _, _, new_pos = new_game_state['self']
        _, old_dist = bfs_direction_and_distance(
            old_pos, old_game_state['coins'], old_game_state['field'],
            old_game_state['bombs'], old_game_state['others'])
        _, new_dist = bfs_direction_and_distance(
            new_pos, new_game_state['coins'], new_game_state['field'],
            new_game_state['bombs'], new_game_state['others'])
        if old_dist is not None and new_dist is not None:
            if new_dist < old_dist:
                events.append(MOVED_CLOSER_TO_COIN)
            elif new_dist > old_dist:
                events.append(MOVED_FARTHER_FROM_COIN)    

    # dense shaping: close in on an opponent, weaker than the coin term so
    # hunting never outbids staying safe
    if old_game_state is not None and new_game_state is not None:
        _, old_opp_dist = bfs_to_opponents(
            old_game_state['self'][3], old_game_state['field'],
            old_game_state['bombs'], old_game_state['others'])
        _, new_opp_dist = bfs_to_opponents(
            new_game_state['self'][3], new_game_state['field'],
            new_game_state['bombs'], new_game_state['others'])
        if old_opp_dist is not None and new_opp_dist is not None:
            if new_opp_dist < old_opp_dist:
                events.append(MOVED_CLOSER_TO_OPPONENT)
            elif new_opp_dist > old_opp_dist:
                events.append(MOVED_FARTHER_FROM_OPPONENT)

    reward = reward_from_events(self, events)
    note_events(self, events, reward)

    if old_f is None or self_action is None:
        return

    # Bootstrapped TD target: 
    # reward now + discounted best value of where we ended up 
    future_q = self.target_model.predict(new_f)
    mask = get_action_mask(new_game_state)
    allowed_future_q = {a: q for a, q in future_q.items() if mask[a]} or future_q
    td_target = reward + self.gamma * max(allowed_future_q.values())

    # before the update, so the curve shows the error the model still had
    self.round_td_abs += abs(td_target - self.model.predict(old_f)[self_action])
    self.round_updates += 1

    self.model.update(old_f, self_action, td_target, self.lr)

# save final obtained model 
def end_of_round(self, last_game_state, last_action, events):
    # P5: shape toward the actual Task-3 goal -- WINNING the round, not just
    # farming. Auxiliary reward, absent in official (non-training) games.
    if last_game_state is not None and last_game_state.get('others'):
        my_score = last_game_state['self'][1]
        if my_score > max(o[1] for o in last_game_state['others']):
            events.append(WIN_ROUND)

    last_f = state_to_features(last_game_state)
    reward = reward_from_events(self, events)
    note_events(self, events, reward)

    if last_f is not None and last_action is not None:
        # Terminal transition
        td_target = reward
        self.model.update(last_f, last_action, td_target, self.lr)

    # final update
    self.model.save("model.npy")

    # saves every 250 rounds a model
    if last_game_state['round'] % 250 == 0:
        self.model.save(f"model_round{last_game_state['round']}.npy")

    write_round_log(self, last_game_state)

    self.recent_positions.clear()
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    self.lr = max(self.lr_min, self.lr * self.lr_decay)

    # THIRD FIX
    if last_game_state['round'] % self.target_sync_every == 0:
        self.target_model = copy.deepcopy(self.model)


def write_round_log(self, last_game_state):
    opp_score = [ o[1] for o in (last_game_state.get('others') or [])]
    opp_max = max(opp_score) if opp_score else 0
    my_score = last_game_state['self'][1]
    win = 1 if my_score > opp_max else 0

    counted = self.round_events
    row = [
        last_game_state["round"],
        last_game_state["step"],
        round(self.round_reward, 3),
        round(self.round_td_abs / self.round_updates, 4) if self.round_updates else 0.0,
        round(self.epsilon, 5),
        round(self.lr, 6),
        counted.get(e.COIN_COLLECTED, 0),
        counted.get(e.CRATE_DESTROYED, 0),
        counted.get(e.BOMB_DROPPED, 0),
        counted.get(e.KILLED_OPPONENT, 0),
        counted.get(e.KILLED_SELF, 0),
        counted.get(e.GOT_KILLED, 0),
        counted.get(e.INVALID_ACTION, 0),
        counted.get(BOMBED_OPPONENT, 0),
        counted.get(TRAPPED_OPPONENT, 0),
        opp_max,
        win,
    ]
    with open(self.log_path, "a", newline="") as fh:
        csv.writer(fh).writerow(row)
    reset_round_log(self)

# keeps score from the current points
def reward_from_events(self, events):
    # the environment adds GOT_KILLED to every death, KILLED_SELF included.
    # counting both would price a suicide at -23 by accident
    if e.GOT_KILLED in events and e.KILLED_SELF in events:
        events = [ev for ev in events if ev != e.GOT_KILLED]

    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 6,
        e.KILLED_SELF: -15,
        e.GOT_KILLED: -8,
        e.INVALID_ACTION: -1,
        e.BOMB_DROPPED: 0.0,  
        e.CRATE_DESTROYED: 0.5, # should it be dropped to 0.25? 
        e.COIN_FOUND: 0.3,
        e.SURVIVED_ROUND: 0.5,
        IN_DANGER: -0.15,       # dense, immediate
        MOVED_CLOSER_TO_COIN: 0.1,
        MOVED_FARTHER_FROM_COIN: -0.1,  # same magnitude
        BOMBED_CRATES: 0.3,
        ESCAPED_DANGER: 0.5,
        STALLED: -0.3,
        # raising these to 1.0 / 2.0 was measured twice and lost both times
        BOMBED_OPPONENT: 0.8,
        TRAPPED_OPPONENT: 2.0, # NEW SIGNAL: a bomb they cannot escape (stacks with BOMBED_OPPONENT above)
        MOVED_CLOSER_TO_OPPONENT: 0.15,
        MOVED_FARTHER_FROM_OPPONENT: -0.15,  # same magnitude, no free farming
        WAITED_WITH_OPPONENTS: -0.1, # 4. FIX
        AMBUSH_READY: 0.3, # P5: bombe ready and rival in blast
        WIN_ROUND: 1.5, # P%: explicitily winning the run 
    }
    STEP_PENALTY = -0.03 # bc coins should be collected as fast as possible
    return STEP_PENALTY + sum(game_rewards.get(ev, 0) for ev in events)