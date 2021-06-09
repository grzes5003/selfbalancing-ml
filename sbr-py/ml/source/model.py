from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tf_agents.environments.utils import validate_py_environment
from .TF_interface import RobotModel


class ML_Model:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

        self.model = None
        self.dqn = None

        self.build_model(self.states, self.actions)
        self.build_agent()
        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    def build_model(self, states, actions):
        self.model = Sequential()
        # self.model.add(Flatten(input_shape=1))
        self.model.add(tf.keras.Input(shape=states))
        # self.model.add(Flatten(input_shape=(1,)))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(actions, activation='linear'))
        print(self.model.summary())
        return self.model

    def build_agent(self):
        if self.model is None:
            print("Warn: model not initiated")
            return
        policy = BoltzmannQPolicy()
        memory = SequentialMemory(limit=50000, window_length=1)
        self.dqn = DQNAgent(model=self.model, memory=memory, policy=policy,
                            nb_actions=self.actions, nb_steps_warmup=10, target_model_update=1e-2)
        return self.dqn

    def fit(self, env, nb_steps):
        if self.dqn is None or self.model is None:
            print("Err: model or dqn is not initialized")
            return
        self.dqn.fit(env, nb_steps=nb_steps, visualize=False, verbose=1)

    @staticmethod
    def validate(env):
        validate_py_environment(env, episodes=5)

