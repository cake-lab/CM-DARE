# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reinforcement learning models and parameters."""

import collections
import functools
import operator
import gym

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import discretization
from tensor2tensor.rl.envs import tf_atari_wrappers
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_hparams
def ppo_base_v1():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.learning_rate = 1e-4
  hparams.add_hparam("init_mean_factor", 0.1)
  hparams.add_hparam("init_logstd", 0.1)
  hparams.add_hparam("policy_layers", (100, 100))
  hparams.add_hparam("value_layers", (100, 100))
  hparams.add_hparam("num_agents", 30)
  hparams.add_hparam("clipping_coef", 0.2)
  hparams.add_hparam("gae_gamma", 0.99)
  hparams.add_hparam("gae_lambda", 0.95)
  hparams.add_hparam("entropy_loss_coef", 0.01)
  hparams.add_hparam("value_loss_coef", 1)
  hparams.add_hparam("optimization_epochs", 15)
  hparams.add_hparam("epoch_length", 200)
  hparams.add_hparam("epochs_num", 2000)
  hparams.add_hparam("eval_every_epochs", 10)
  hparams.add_hparam("num_eval_agents", 3)
  hparams.add_hparam("video_during_eval", False)
  hparams.add_hparam("save_models_every_epochs", 30)
  hparams.add_hparam("optimization_batch_size", 50)
  hparams.add_hparam("max_gradients_norm", 0.5)
  hparams.add_hparam("simulation_random_starts", False)
  hparams.add_hparam("simulation_flip_first_random_for_beginning", False)
  hparams.add_hparam("intrinsic_reward_scale", 0.)
  hparams.add_hparam("logits_clip", 4.0)
  hparams.add_hparam("dropout_ppo", 0.1)
  return hparams


@registry.register_hparams
def ppo_continuous_action_base():
  hparams = ppo_base_v1()
  hparams.add_hparam("policy_network", feed_forward_gaussian_fun)
  hparams.add_hparam("policy_network_params", "basic_policy_parameters")
  return hparams


@registry.register_hparams
def basic_policy_parameters():
  wrappers = None
  return tf.contrib.training.HParams(wrappers=wrappers)


@registry.register_hparams
def ppo_discrete_action_base():
  hparams = ppo_base_v1()
  hparams.add_hparam("policy_network", feed_forward_categorical_fun)
  return hparams


@registry.register_hparams
def discrete_random_action_base():
  hparams = common_hparams.basic_params1()
  hparams.add_hparam("policy_network", random_policy_fun)
  return hparams


@registry.register_hparams
def ppo_atari_base():
  """Atari base parameters."""
  hparams = ppo_discrete_action_base()
  hparams.learning_rate = 4e-4
  hparams.num_agents = 5
  hparams.epoch_length = 200
  hparams.gae_gamma = 0.985
  hparams.gae_lambda = 0.985
  hparams.entropy_loss_coef = 0.002
  hparams.value_loss_coef = 0.025
  hparams.optimization_epochs = 10
  hparams.epochs_num = 10000
  hparams.num_eval_agents = 1
  hparams.network = feed_forward_cnn_small_categorical_fun
  return hparams


@registry.register_hparams
def ppo_pong_base():
  """Pong base parameters."""
  hparams = ppo_discrete_action_base()
  hparams.learning_rate = 1e-4
  hparams.num_agents = 8
  hparams.epoch_length = 200
  hparams.gae_gamma = 0.985
  hparams.gae_lambda = 0.985
  hparams.entropy_loss_coef = 0.003
  hparams.value_loss_coef = 1
  hparams.optimization_epochs = 3
  hparams.epochs_num = 1000
  hparams.num_eval_agents = 1
  hparams.policy_network = feed_forward_cnn_small_categorical_fun
  hparams.clipping_coef = 0.2
  hparams.optimization_batch_size = 20
  hparams.max_gradients_norm = 0.5
  return hparams


def simple_gym_spec(env):
  """Parameters of environment specification."""
  standard_wrappers = None
  env_lambda = None
  if isinstance(env, str):
    env_lambda = lambda: gym.make(env)
  if callable(env):
    env_lambda = env
  assert env_lambda is not None, "Unknown specification of environment"

  return tf.contrib.training.HParams(env_lambda=env_lambda,
                                     wrappers=standard_wrappers,
                                     simulated_env=False)


def standard_atari_env_spec(
    env=None, simulated=False, resize_height_factor=1, resize_width_factor=1,
    grayscale=False, include_clipping=True, batch_env=None):
  """Parameters of environment specification."""
  resize_wrapper = [tf_atari_wrappers.ResizeWrapper,
                    {"height_factor": resize_height_factor,
                     "width_factor": resize_width_factor,
                     "grayscale": grayscale}]
  if include_clipping:
    standard_wrappers = [
        resize_wrapper,
        [tf_atari_wrappers.RewardClippingWrapper, {}],
        [tf_atari_wrappers.StackWrapper, {"history": 4}],
    ]
  else:
    standard_wrappers = [
        resize_wrapper,
        [tf_atari_wrappers.StackWrapper, {"history": 4}],
    ]
  if simulated:  # No resizing on simulated environments.
    standard_wrappers = standard_wrappers[1:]

  env_spec = tf.contrib.training.HParams(
      wrappers=standard_wrappers,
      simulated_env=simulated)

  if batch_env is not None:
    env_spec.add_hparam("batch_env", batch_env)
  else:
    env_lambda = None
    if isinstance(env, str):
      env_lambda = lambda: gym.make(env)
    if callable(env):
      env_lambda = env
    assert env_lambda is not None, "Unknown specification of environment"
    env_spec.add_hparam("env_lambda", env_lambda)

  return env_spec


def standard_atari_env_simulated_spec(
    real_env, video_num_input_frames, video_num_target_frames):
  """Spec."""
  env_spec = standard_atari_env_spec(
      # This hack is here because SimulatedBatchEnv needs to get
      # observation_space from the real env. TODO(koz4k): refactor.
      env=lambda: real_env,
      simulated=True
  )
  env_spec.add_hparam("simulation_random_starts", True)
  env_spec.add_hparam("simulation_flip_first_random_for_beginning", True)
  env_spec.add_hparam("intrinsic_reward_scale", 0.0)
  env_spec.add_hparam("initial_frames_problem", real_env)
  env_spec.add_hparam("video_num_input_frames", video_num_input_frames)
  env_spec.add_hparam("video_num_target_frames", video_num_target_frames)
  return env_spec


def standard_atari_env_eval_spec(
    env, simulated=False, resize_height_factor=1, resize_width_factor=1,
    grayscale=False):
  """Parameters of environment specification for eval."""
  return standard_atari_env_spec(
      env, simulated, resize_height_factor, resize_width_factor, grayscale,
      include_clipping=False)


def standard_atari_ae_env_spec(env, ae_hparams_set):
  """Parameters of environment specification."""
  standard_wrappers = [[tf_atari_wrappers.AutoencoderWrapper,
                        {"ae_hparams_set": ae_hparams_set}],
                       [tf_atari_wrappers.StackWrapper, {"history": 4}]]
  env_lambda = None
  if isinstance(env, str):
    env_lambda = lambda: gym.make(env)
  if callable(env):
    env_lambda = env
  assert env is not None, "Unknown specification of environment"

  return tf.contrib.training.HParams(env_lambda=env_lambda,
                                     wrappers=standard_wrappers,
                                     simulated_env=False)


@registry.register_hparams
def ppo_pong_ae_base():
  """Pong autoencoder base parameters."""
  hparams = ppo_pong_base()
  hparams.learning_rate = 1e-4
  hparams.network = dense_bitwise_categorical_fun
  return hparams


@registry.register_hparams
def pong_model_free():
  """TODO(piotrmilos): Document this."""
  hparams = tf.contrib.training.HParams(
      epochs_num=4,
      eval_every_epochs=2,
      num_agents=10,
      optimization_epochs=3,
      epoch_length=30,
      entropy_loss_coef=0.003,
      learning_rate=8e-05,
      optimizer="Adam",
      policy_network=feed_forward_cnn_small_categorical_fun,
      gae_lambda=0.985,
      num_eval_agents=1,
      max_gradients_norm=0.5,
      gae_gamma=0.985,
      optimization_batch_size=4,
      clipping_coef=0.2,
      value_loss_coef=1,
      save_models_every_epochs=False)
  hparams.add_hparam("environment_spec",
                     standard_atari_env_spec("PongNoFrameskip-v4"))
  hparams.add_hparam(
      "environment_eval_spec",
      standard_atari_env_eval_spec("PongNoFrameskip-v4"))
  return hparams


NetworkOutput = collections.namedtuple(
    "NetworkOutput", "policy, value, action_postprocessing")


def feed_forward_gaussian_fun(action_space, config, observations):
  """Feed-forward Gaussian."""
  if not isinstance(action_space, gym.spaces.box.Box):
    raise ValueError("Expecting continuous action space.")

  mean_weights_initializer = tf.contrib.layers.variance_scaling_initializer(
      factor=config.init_mean_factor)
  logstd_initializer = tf.random_normal_initializer(config.init_logstd, 1e-10)

  flat_observations = tf.reshape(observations, [
      tf.shape(observations)[0], tf.shape(observations)[1],
      functools.reduce(operator.mul, observations.shape.as_list()[2:], 1)])

  with tf.variable_scope("network_parameters"):
    with tf.variable_scope("policy"):
      x = flat_observations
      for size in config.policy_layers:
        x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
      mean = tf.contrib.layers.fully_connected(
          x, action_space.shape[0], tf.tanh,
          weights_initializer=mean_weights_initializer)
      logstd = tf.get_variable(
          "logstd", mean.shape[2:], tf.float32, logstd_initializer)
      logstd = tf.tile(
          logstd[None, None],
          [tf.shape(mean)[0], tf.shape(mean)[1]] + [1] * (mean.shape.ndims - 2))
    with tf.variable_scope("value"):
      x = flat_observations
      for size in config.value_layers:
        x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
      value = tf.contrib.layers.fully_connected(x, 1, None)[..., 0]
  mean = tf.check_numerics(mean, "mean")
  logstd = tf.check_numerics(logstd, "logstd")
  value = tf.check_numerics(value, "value")

  policy = tf.contrib.distributions.MultivariateNormalDiag(mean,
                                                           tf.exp(logstd))

  return NetworkOutput(policy, value, lambda a: tf.clip_by_value(a, -2., 2))


def clip_logits(logits, config):
  logits_clip = getattr(config, "logits_clip", 0.)
  if logits_clip > 0:
    min_logit = tf.reduce_min(logits)
    return tf.minimum(logits - min_logit, logits_clip)
  else:
    return logits


def feed_forward_categorical_fun(action_space, config, observations):
  """Feed-forward categorical."""
  if not isinstance(action_space, gym.spaces.Discrete):
    raise ValueError("Expecting discrete action space.")
  flat_observations = tf.reshape(observations, [
      tf.shape(observations)[0], tf.shape(observations)[1],
      functools.reduce(operator.mul, observations.shape.as_list()[2:], 1)])
  with tf.variable_scope("network_parameters"):
    with tf.variable_scope("policy"):
      x = flat_observations
      for size in config.policy_layers:
        x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
      logits = tf.contrib.layers.fully_connected(x, action_space.n,
                                                 activation_fn=None)
    with tf.variable_scope("value"):
      x = flat_observations
      for size in config.value_layers:
        x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
      value = tf.contrib.layers.fully_connected(x, 1, None)[..., 0]
  logits = clip_logits(logits, config)
  policy = tf.contrib.distributions.Categorical(logits=logits)
  return NetworkOutput(policy, value, lambda a: a)


def feed_forward_cnn_small_categorical_fun(action_space, config, observations):
  """Small cnn network with categorical output."""
  obs_shape = common_layers.shape_list(observations)
  x = tf.reshape(observations, [-1] + obs_shape[2:])
  with tf.variable_scope("network_parameters"):
    dropout = getattr(config, "dropout_ppo", 0.0)
    with tf.variable_scope("feed_forward_cnn_small"):
      x = tf.to_float(x) / 255.0
      x = tf.contrib.layers.conv2d(x, 32, [5, 5], [2, 2],
                                   activation_fn=tf.nn.relu, padding="SAME")
      x = tf.contrib.layers.conv2d(x, 32, [5, 5], [2, 2],
                                   activation_fn=tf.nn.relu, padding="SAME")

      flat_x = tf.reshape(
          x, [obs_shape[0], obs_shape[1],
              functools.reduce(operator.mul, x.shape.as_list()[1:], 1)])
      flat_x = tf.nn.dropout(flat_x, keep_prob=1.0 - dropout)
      x = tf.contrib.layers.fully_connected(flat_x, 128, tf.nn.relu)

      logits = tf.contrib.layers.fully_connected(x, action_space.n,
                                                 activation_fn=None)
      logits = clip_logits(logits, config)

      value = tf.contrib.layers.fully_connected(
          x, 1, activation_fn=None)[..., 0]
      policy = tf.contrib.distributions.Categorical(logits=logits)
  return NetworkOutput(policy, value, lambda a: a)


def feed_forward_cnn_small_categorical_fun_new(
    action_space, config, observations):
  """Small cnn network with categorical output."""
  obs_shape = common_layers.shape_list(observations)
  x = tf.reshape(observations, [-1] + obs_shape[2:])
  with tf.variable_scope("network_parameters"):
    dropout = getattr(config, "dropout_ppo", 0.0)
    with tf.variable_scope("feed_forward_cnn_small"):
      x = tf.to_float(x) / 255.0
      x = tf.nn.dropout(x, keep_prob=1.0 - dropout)
      x = tf.layers.conv2d(
          x, 32, (4, 4), strides=(2, 2), name="conv1",
          activation=common_layers.belu, padding="SAME")
      x = tf.nn.dropout(x, keep_prob=1.0 - dropout)
      x = tf.layers.conv2d(
          x, 64, (4, 4), strides=(2, 2), name="conv2",
          activation=common_layers.belu, padding="SAME")
      x = tf.nn.dropout(x, keep_prob=1.0 - dropout)
      x = tf.layers.conv2d(
          x, 128, (4, 4), strides=(2, 2), name="conv3",
          activation=common_layers.belu, padding="SAME")

      flat_x = tf.reshape(
          x, [obs_shape[0], obs_shape[1],
              functools.reduce(operator.mul, x.shape.as_list()[1:], 1)])
      flat_x = tf.nn.dropout(flat_x, keep_prob=1.0 - dropout)
      x = tf.layers.dense(flat_x, 128, activation=tf.nn.relu, name="dense1")

      logits = tf.layers.dense(x, action_space.n, name="dense2")
      logits = clip_logits(logits, config)

      value = tf.layers.dense(x, 1, name="value")[..., 0]
      policy = tf.contrib.distributions.Categorical(logits=logits)

  return NetworkOutput(policy, value, lambda a: a)


def dense_bitwise_categorical_fun(action_space, config, observations):
  """Dense network with bitwise input and categorical output."""
  del config
  obs_shape = common_layers.shape_list(observations)
  x = tf.reshape(observations, [-1] + obs_shape[2:])

  with tf.variable_scope("network_parameters"):
    with tf.variable_scope("dense_bitwise"):
      x = discretization.int_to_bit_embed(x, 8, 32)
      flat_x = tf.reshape(
          x, [obs_shape[0], obs_shape[1],
              functools.reduce(operator.mul, x.shape.as_list()[1:], 1)])

      x = tf.contrib.layers.fully_connected(flat_x, 256, tf.nn.relu)
      x = tf.contrib.layers.fully_connected(flat_x, 128, tf.nn.relu)

      logits = tf.contrib.layers.fully_connected(x, action_space.n,
                                                 activation_fn=None)

      value = tf.contrib.layers.fully_connected(
          x, 1, activation_fn=None)[..., 0]
      policy = tf.contrib.distributions.Categorical(logits=logits)

  return NetworkOutput(policy, value, lambda a: a)


def random_policy_fun(action_space, unused_config, observations):
  """Random policy with categorical output."""
  obs_shape = observations.shape.as_list()
  with tf.variable_scope("network_parameters"):
    value = tf.zeros(obs_shape[:2])
    policy = tf.distributions.Categorical(
        probs=[[[1. / float(action_space.n)] * action_space.n
               ] * (obs_shape[0] * obs_shape[1])])
  return NetworkOutput(policy, value, lambda a: a)
