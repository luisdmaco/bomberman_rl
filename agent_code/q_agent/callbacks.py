from .features import state_to_features, get_action_mask
from .q_model import LinearQModel
import numpy as np

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
FEATURE_NAMES = [
    'walkable_up', 'walkable_down', 'walkable_left', 'walkable_right',
    'danger_here', 'danger_up', 'danger_down', 'danger_left', 'danger_right',
    'can_bomb',
    'coin_dir_up', 'coin_dir_down', 'coin_dir_left', 'coin_dir_right', 'coin_dir_none',
    'coin_dist',    
    'crate_dir_up', 'crate_dir_down', 'crate_dir_left', 'crate_dir_right', 'crate_dir_none',
    'safe_dir_up', 'safe_dir_down', 'safe_dir_left', 'safe_dir_right', 'safe_dir_none',
    'crates_in_blast',
    'opp_dir_up', 'opp_dir_down', 'opp_dir_left', 'opp_dir_right', 'opp_dir_none',
    'opp_dist',
    'opp_in_blast',
]
N_FEATURES = len(FEATURE_NAMES)

def setup(self):
    self.model = LinearQModel(n_features=N_FEATURES, actions=ACTIONS)
    if not self.train:
        self.model.load("model.npy")
        # a checkpoint from an older feature set would otherwise die inside
        # the first dot product
        saved = len(next(iter(self.model.weights.values())))
        if saved != N_FEATURES:
            raise ValueError(
                f"model.npy holds {saved} weights per action but the feature "
                f"vector has {N_FEATURES} entries. This checkpoint predates the "
                f"current features.py -- retrain or restore a matching model.")

def act(self, game_state):
    features = state_to_features(game_state)
    q_values = self.model.predict(features)
    mask = get_action_mask(game_state)
    allowed = [a for a in ACTIONS if mask[a]]

    if self.train and np.random.rand() < self.epsilon:
        action = np.random.choice(allowed)
        self.logger.debug(f"Exploring: random action {action} (allowed={allowed})")
        return action

    allowed_q = {a: q for a, q in q_values.items() if mask[a]}
    action = max(allowed_q, key=allowed_q.get)
    self.logger.debug(f"Exploiting: Q-values {q_values}, mask={mask}, chose {action}")
    return action