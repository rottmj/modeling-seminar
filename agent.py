import numpy as np


class Agent:
    def __init__(self, model) -> None:
        if model == 'linear':
            # 19 weights + 1 bias
            self.thetas = np.zeros(shape=(20,))
        else:
            pass


    def update(self, deltas, rewards, step_size, nb_steps):
        differnce_rewards = ...
        self.theta += step_size/nb_steps * np.sum(differnce_rewards, deltas)


    def sample_deltas(self, nb_directions):
        # Methode auf Korrektheit überprüfen
        return [np.random.rand(self.theta.shape) for _ in range(nb_directions)]


    def compute_action(self, state):
        return np.dot(self.thetas[:-2], state) + self.thetas[-1]
    
    
    def select_action(self):
        # findet bisher in environment statt
        pass


    def train():
        # model trainieren
        pass