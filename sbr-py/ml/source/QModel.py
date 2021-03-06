import math
import numpy as np


class QModel:
    """Main model used to learn Robot"""
    def __init__(self, n_bins, action_space):
        self.q_table = np.zeros(n_bins + (action_space,))
        print(f'q_table shape {self.q_table.shape}')        # shape = (60, 6, 5)

    def policy(self, state: tuple):
        """Choosing action based on epsilon-greedy policy"""
        return np.argmax(self.q_table[state])

    def new_q_value(self, reward: float, new_state: tuple, discount_factor=1) -> float:
        """Temperal diffrence for updating Q-value of state-action pair"""
        future_optimal_value = np.max(self.q_table[new_state])
        learned_value = reward + discount_factor * future_optimal_value
        return learned_value

    def set_q_table(self, q_table: np.ndarray):
        self.q_table = q_table

    @staticmethod
    def learning_rate(n: int, min_rate=0.01) -> float:
        """Decaying learning rate"""
        return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 6)))

    @staticmethod
    def exploration_rate(n: int, min_rate=0.1) -> float:
        """Decaying exploration rate"""
        return max(min_rate, min(1, 1.0 - math.log10((n + 1) / 25)))
