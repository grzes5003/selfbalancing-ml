from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
from datetime import datetime

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from .robot_interface import RobotInterface

episode_time_limit = 5
tolerance = 1.5
swing_tolerance_limit = 15


class RobotModel(py_environment.PyEnvironment):
    def __init__(self):
        super().__init__()
        self._robot = RobotInterface()
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=4, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=np.int32, minimum=-255, maximum=255,
                                                             name='observation')
        time.sleep(0.5)
        self._state = self._robot.getState()
        self._episode_timer = time.time()
        self._episode_limit = episode_time_limit
        self._zero = self._robot.ZERO
        self._start_time = time.time()
        self._tolerance = tolerance
        self.done = False
        self._sleep_interval = 0.25

    def observation_spec(self) -> types.NestedArraySpec:
        return self._observation_spec

    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec

    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        self._episode_timer = time.time()
        self._state = self._robot.getState()
        if self._episode_timer - self._start_time >= self._episode_limit \
                or abs(self._state) > swing_tolerance_limit:
            self.done = True
            return self.reset()

        self._robot.setState(action)
        time.sleep(self._sleep_interval)

        if self._zero + self._tolerance >= self._state >= self._zero - self._tolerance:
            return ts.termination(np.array([self._state], dtype=np.int32), reward=1.0)
        else:
            return ts.transition(np.array([self._state], dtype=np.int32), reward=-1.0)

    def _reset(self) -> ts.TimeStep:
        print("{} _reset call".format(datetime.now()))
        # self._state = self._robot.ZERO
        _timer = time.time()
        while True:
            time.sleep(0.1)
            self._state = self._robot.getState()
            if abs(self._zero) + self._tolerance <= abs(self._state) or abs(self._state) <= abs(self._zero) - self._tolerance:
                _timer = time.time()
            if abs(self._zero) + self._tolerance >= abs(self._state) >= abs(self._zero) - self._tolerance \
                    and time.time() - _timer > 2:
                print("{} _reset done".format(datetime.now()))
                break
        self._start_time = time.time()
        return ts.restart(np.array([self._state], dtype=np.int32))

    def close_(self):
        self._robot.stop()
