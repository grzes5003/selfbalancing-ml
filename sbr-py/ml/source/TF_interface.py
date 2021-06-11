from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
from datetime import datetime

from sklearn.preprocessing import KBinsDiscretizer
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
import tensorflow as tf

from .robot_interface import RobotInterface

episode_time_limit = 5
raw_tolerance = 1.5
swing_tolerance_limit = 6
NUM_OF_STATES = 60


class RobotModel(py_environment.PyEnvironment):
    def __init__(self):
        super().__init__()
        self._robot = RobotInterface()
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int8, minimum=0, maximum=4,
                                                        name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(1, 1), dtype=np.int8, minimum=0, maximum=127,
                                                             name='observation')
        time.sleep(1)
        self._robot.setState(np.array([2], np.int8))
        self._episode_timer = time.time()
        self._episode_time_limit = episode_time_limit

        self._raw_min_range = self._robot.RAW_MIN_RANGE
        self._raw_max_range = self._robot.RAW_MAX_RANGE

        self.est = KBinsDiscretizer(n_bins=(NUM_OF_STATES, 6), encode='ordinal', strategy='uniform')
        lower_bounds = [self._raw_min_range, -5]
        upper_bounds = [self._raw_max_range, 5]
        self.est.fit([lower_bounds, upper_bounds])

        self._zero = self._discretizer(self._robot.RAW_ZERO)[0]
        self._state = self._discretizer(*self._robot.getState())
        self._start_time = time.time()
        self._upper_tolerance = self._discretizer(self._robot.RAW_ZERO + raw_tolerance)[0]
        self._lower_tolerance = self._discretizer(self._robot.RAW_ZERO - raw_tolerance)[0]
        self._upper_swing_tolerance_limit = self._discretizer(self._robot.RAW_ZERO + swing_tolerance_limit)[0]
        self._lower_swing_tolerance_limit = self._discretizer(self._robot.RAW_ZERO - swing_tolerance_limit)[0]
        self.done = False
        self._sleep_interval = 0.1
        # self._robot.setState(np.array([0]))

    def observation_spec(self) -> types.NestedArraySpec:
        return self._observation_spec

    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec

    def _discretizer(self, gyro, acc=1):
        """Convert continues state intro a discrete state"""
        return tuple(map(int, self.est.transform([[gyro, acc]])[0]))

    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        self._episode_timer = time.time()
        self._state = self._discretizer(*self._robot.getState())

        if self.done:
            return self.reset()

        if self._episode_timer - self._start_time >= self._episode_time_limit:
            self.done = True
            return ts.termination(np.array([self._state], dtype=np.int8), reward=100.0)

        if self._upper_swing_tolerance_limit <= self._state[0] \
                or self._lower_swing_tolerance_limit >= self._state[0]:
            self.done = True
            return ts.termination(np.array([self._state], dtype=np.int8), reward=-100.0)

        print("setState: {}".format(action))
        self._robot.setState(np.array([action], np.int8))
        time.sleep(self._sleep_interval)
        # TODO change rewards scheme
        if self._upper_tolerance >= self._state[0] >= self._lower_tolerance:
            return ts.transition(np.array([self._state], dtype=np.int8), reward=20.0)
        else:
            reward_val = -1.0 - int(self._state[0] / 30) \
                         - (30 if 5 <= self._state[1] or self._state[1] <= 1 else 0) \
                         + int(self._episode_timer - self._start_time)
            return ts.transition(np.array([self._state], dtype=np.int8), reward=reward_val)

    def _reset(self) -> ts.TimeStep:
        print("{} _reset call".format(datetime.now()))
        self._robot.setState(np.array([2], np.int8))
        # self._state = self._robot.ZERO
        _timer = time.time()
        while True:
            time.sleep(0.1)
            self._state = self._discretizer(*self._robot.getState())
            if self._upper_tolerance <= self._state[0] or self._state[0] <= self._lower_tolerance:
                _timer = time.time()
            if self._upper_tolerance >= abs(self._state[0]) >= self._lower_tolerance \
                    and time.time() - _timer > 2:
                print("{} _reset done".format(datetime.now()))
                break
        self._start_time = time.time()
        print("{} in _reset {}".format(datetime.now(), np.array([self._state], dtype=np.float16)))
        return ts.restart(np.array([self._state], dtype=np.int8))

    def close_(self):
        self._robot.setState(np.array([2], np.int8))
        self._robot.stop()
