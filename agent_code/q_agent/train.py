from .features import state_to_features, danger_map, get_action_mask, bfs_direction_and_distance
import events as e

IN_DANGER = "IN_DANGER"
MOVED_CLOSER_TO_COIN = "MOVED_CLOSER_TO_COIN"
MOVED_FARTHER_FROM_COIN = "MOVED_FARTHER_FROM_COIN"

# TODO: are these the best values for our case?
# Hyperparameter training
def setup_training(self):
    self.epsilon = 0.2
    self.epsilon_min = 0.05
    self.epsilon_decay = 0.997 # multiply after every round
    self.lr = 0.01
    self.gamma = 0.9

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
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# keeps score from the current points
def reward_from_events(self, events):
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -5,
        e.INVALID_ACTION: -1,
        e.BOMB_DROPPED: -0.2,  # small cost: must be earned later
        IN_DANGER: -0.3,       # dense, immediate
        MOVED_CLOSER_TO_COIN: 0.1,
        MOVED_FARTHER_FROM_COIN: -0.1,  # same magnitude
    }
    STEP_PENALTY = -0.1 # bc coins should be collectes as fast as possible
    return STEP_PENALTY + sum(game_rewards.get(ev, 0) for ev in events)