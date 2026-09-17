# Keeps the q-learning logic

import numpy as np 


class LinearQModel:
    def __init__(self, n_features, actions):
        self.actions = actions
        self.weights = {a: np.zeros(n_features) for a in actions}

    def predict(self, features):
        return {a: self.weights[a] @ features for a in self.actions}

    #  FIRST FIX: max_norm from our checkpoints
    def update(self, features, action, td_target, lr, max_norm=10.0):
        pred = self.weights[action] @ features
        self.weights[action] += lr * (td_target - pred) * features

        # hard ceiling: catches TD divergence on *any* action
        # without needing to predict in advance which action's Q will run away
        norm = np.linalg.norm(self.weights[action])
        if norm > max_norm:
            self.weights[action] *= max_norm / norm

    def save(self, path):
        np.save(path, self.weights)

    def load(self, path):
        self.weights = np.load(path, allow_pickle=True). item()

        