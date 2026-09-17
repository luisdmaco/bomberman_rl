# Keeps the q-learning logic

import numpy as np 


class LinearQModel:
    def __init__(self, n_features, actions):
        self.actions = actions
        self.weights = {a: np.zeros(n_features) for a in actions}

    def predict(self, features):
        return {a: self.weights[a] @ features for a in self.actions}

    def update(self, features, action, td_target, lr):
        pred = self.weights[action] @ features
        self.weights[action] += lr * (td_target - pred) * features

    def save(self, path):
        np.save(path, self.weights)

    def load(self, path):
        self.weights = np.load(path, allow_pickle=True). item()

        