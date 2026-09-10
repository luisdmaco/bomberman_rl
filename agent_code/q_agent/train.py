from .features import state_to_features, danger_map, get_action_mask, bfs_direction_and_distance, get_blast_coords
import events as e
from collections import deque

IN_DANGER = "IN_DANGER"
MOVED_CLOSER_TO_COIN = "MOVED_CLOSER_TO_COIN"
MOVED_FARTHER_FROM_COIN = "MOVED_FARTHER_FROM_COIN"
BOMBED_CRATES = "BOMBED_CRATES"
ESCAPED_DANGER = "ESCAPED_DANGER"
STALLED = "STALLED"

# TODO: are these the best values for our case?
# Hyperparameter training
def setup_training(self):
    self.epsilon = 0.2
    self.epsilon_min = 0.05
    self.epsilon_decay = 0.9995 # multiply after every round
    self.lr = 0.01
    self.lr_min = 0.001
    self.lr_decay = 0.9995
    self.gamma = 0.95

    self.recent_positions = deque(maxlen=10)

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
        n_crates = sum(1 for (bx, by) in get_blast_coords(ox, oy, f) if f[bx, by] == 1)
        for _ in range(n_crates):
            events.append(BOMBED_CRATES) # once per crate -> reward scales with payoff

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

    reward = reward_from_events(self, events)

    if old_f is None or self_action is None:
        return

    # Bootstrapped TD target: 
    # reward now + discounted best value of where we ended up 
    future_q = self.model.predict(new_f)
    mask = get_action_mask(new_game_state)
    allowed_future_q = {a: q for a, q in future_q.items() if mask[a]} or future_q
    td_target = reward + self.gamma * max(allowed_future_q.values())

    self.model.update(old_f, self_action, td_target, self.lr)

# save final obtained model 
def end_of_round(self, last_game_state, last_action, events):
    last_f = state_to_features(last_game_state)
    reward = reward_from_events(self, events)

    if last_f is not None and last_action is not None:
        # Terminal transition
        td_target = reward
        self.model.update(last_f, last_action, td_target, self.lr)

    # final update
    self.model.save("model.npy")

    # saves every 250 rounds a model
    if last_game_state['round'] % 250 == 0:
        self.model.save(f"model_round{last_game_state['round']}.npy")

    self.recent_positions.clear()
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    self.lr = max(self.lr_min, self.lr * self.lr_decay)

# keeps score from the current points
def reward_from_events(self, events):
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -15,
        e.INVALID_ACTION: -1,
        e.BOMB_DROPPED: 0.0,  
        e.CRATE_DESTROYED: 0.5, # should it be dropped to 0.25? 
        e.COIN_FOUND: 0.3,
        e.SURVIVED_ROUND: 0.5,
        IN_DANGER: -0.05,       # dense, immediate
        MOVED_CLOSER_TO_COIN: 0.1,
        MOVED_FARTHER_FROM_COIN: -0.1,  # same magnitude
        BOMBED_CRATES: 0.3,
        ESCAPED_DANGER: 0.4,
        STALLED: -0.3,
    }
    STEP_PENALTY = -0.02 # bc coins should be collected as fast as possible
    return STEP_PENALTY + sum(game_rewards.get(ev, 0) for ev in events)