import numpy as np
import matplotlib.pyplot as plt
PREFIX = '../ml/tests/tmp/'


# noinspection PyTupleAssignmentBalance
def test_plot_reward(file_name='16231843_result.npz'):
    """Plots scores from file_name"""
    with np.load(PREFIX + file_name) as data:
        scores = data['scores']
        x = range(len(scores))
        m, b = np.polyfit(x, scores, 1)
        plt.plot(x, scores, marker='o')
        plt.plot(x, m * x + b)
        plt.title(file_name)
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.show()
