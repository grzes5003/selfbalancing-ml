import threading
import time
from unittest import TestCase
from unittest.mock import patch, MagicMock
from ml.source import RobotInterface, RobotModel
from ml.source.QModel import QModel
from ml.source.model import ML_Model
from util.connectivity import Connectivity
import numpy as np
from datetime import datetime
import signal
import os
import sys


class TestRobotInterface(TestCase):

    @patch('serial.Serial', autospec=True)
    @patch.object(Connectivity, 'write', MagicMock(return_value=-1.45))
    @patch.object(RobotInterface, 'getState', MagicMock(return_value=-1.45))
    @patch.object(RobotInterface, '_update', MagicMock(return_value=None))
    def test_basic(self, mock_serial):
        env = RobotModel()
        episodes = 5
        for episode in range(1, episodes + 1):
            print("--------------------------")
            env.done = False
            score = 0
            while not env.done:
                time_step = env.step(np.random.randint(5, size=1))
                print(time_step)
                score += time_step.reward
            print('Episode:{} Score:{};'.format(episode, score))
            self.assertEqual(score, 50)
        env.close_()

    @patch('serial.Serial', autospec=True)
    @patch.object(Connectivity, 'write', MagicMock(return_value=-1.45))
    @patch.object(RobotInterface, 'getState', MagicMock(return_value=-1.46))        # <-8, 1.5, 12> ->  <0, 0.52, 1>
    @patch.object(RobotInterface, '_update', MagicMock(return_value=None))
    def test_model_fit(self, mock_model):
        env = RobotModel()
        ml_model = ML_Model(env.observation_spec().shape,
                            env.action_spec().num_values, env)
        ml_model.fit(env, nb_steps=50)
        # ml_model.validate(env)
        env.close_()

    @patch('serial.Serial', autospec=True)
    @patch.object(Connectivity, 'write', MagicMock(return_value=-1.45))
    @patch.object(RobotInterface, 'getState', MagicMock(return_value=-1.46))        # <-8, 1.5, 12> ->  <0, 0.52, 1>
    @patch.object(RobotInterface, '_update', MagicMock(return_value=None))
    def test_model_qlearn(self, mock_model):
        env = RobotModel()
        ml_model = QModel(12, 5)

        n_episodes = 5
        for e in range(n_episodes):
            print("--------------------------")
            current_state, env.done = 6, False

            while not env.done:
                action = ml_model.policy(current_state)

                if np.random.random() < ml_model.exploration_rate(e):
                    action = 2  # TODO random

                obs, reward, done, _ = env.step(action)
                new_state = obs

                # Update Q-Table
                lr = ml_model.learning_rate(e)
                learnt_value = ml_model.new_q_value(reward, new_state)
                old_value = ml_model.q_table[current_state][action]
                ml_model.q_table[current_state][action] = (1 - lr) * old_value + lr * learnt_value

                current_state = new_state
        print(ml_model.q_table)
        env.close_()

    def test_write_real(self):
        env = RobotModel()
        episodes = 5
        for episode in range(1, episodes + 1):
            print("--------------------------")
            env.done = False
            score = 0
            while not env.done:
                time_step = env.step(np.random.randint(5, size=1))
                score += time_step.reward
            print('{} Episode:{} Score:{};'.format(datetime.now(), episode, score))
        env.close_()

    @patch('serial.Serial', autospec=True)
    @patch.object(Connectivity, 'write', MagicMock(return_value=8.1))
    @patch.object(RobotInterface, 'getState', MagicMock(return_value=5))
    @patch.object(RobotInterface, '_update', MagicMock(return_value=None))
    def test_infinite_reset(self, mock_serial):
        env = RobotModel()
        with self.assertRaises(TestTimeout):
            with test_timeout(10):
                _ = env.step(np.random.randint(5, size=1))

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass


class TestTimeout(Exception):
    pass


class test_timeout(threading.Thread):
    def __init__(self, seconds, error_message=None):
        super().__init__()
        if error_message is None:
            error_message = 'test timed out after {}s.'.format(seconds)
        self.seconds = seconds
        self.error_message = error_message
        self._stop_event = threading.Event()

    def handle_timeout(self):
        print("called timeout")
        raise TestTimeout(self.error_message)

    def __enter__(self):
        threading.Thread.__init__(self)
        self._timer = time.time()
        self.setDaemon(True)
        self.start()

    def run(self) -> None:
        print("started")
        while not self._stop_event.is_set():
            if time.time() - self._timer >= self.seconds:
                self.handle_timeout()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
