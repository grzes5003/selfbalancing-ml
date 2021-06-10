from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.environments.utils import validate_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from .TF_interface import RobotModel


class ML_Model:
    def __init__(self, py_env):
        self.model = None
        self.dqn = None
        self.q_net = None
        self.agent = None
        self.env = tf_py_environment.TFPyEnvironment(py_env)
        self.train_step_counter = tf.Variable(0)

        self.action_tensor_spec = tensor_spec.from_spec(self.env.action_spec())
        self.num_actions = self.action_tensor_spec.maximum - self.action_tensor_spec.minimum + 1
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        self.build_model()
        self.build_agent()

    def build_model(self):
        fc_layer_params = (100, 50)
        dense_layers = [ML_Model.dense_layer(num_units) for num_units in fc_layer_params]
        q_values_layer = tf.keras.layers.Dense(
            self.num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2))
        self.q_net = Sequential(dense_layers + [q_values_layer])
        print(self.q_net.summary())
        return self.q_net

    @staticmethod
    def dense_layer(num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal'))

    def build_agent(self):
        if self.q_net is None:
            print("Warn: q_net not initiated")
            return

        memory = SequentialMemory(limit=50000, window_length=1)
        policy = random_tf_policy.RandomTFPolicy(self.env.time_step_spec(), self.env.action_spec())

        self.agent = dqn_agent.DqnAgent(
            self.env.time_step_spec(),
            self.env.train_env.action_spec(),
            q_network=self.q_net,
            optimizer=self.optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.train_step_counter)

        self.agent.initialize()
        return self.agent

    def fit(self, env, nb_steps):
        if self.dqn is None or self.model is None:
            print("Err: model or dqn is not initialized")
            return
        self.dqn.fit(env, nb_steps=nb_steps, visualize=False, verbose=1)

    @staticmethod
    def validate(env):
        validate_py_environment(env, episodes=5)

