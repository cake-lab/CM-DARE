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

"""Definitions of data generators for gym problems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We need gym_utils for the game environments defined there.
from tensor2tensor.data_generators import gym_utils  # pylint: disable=unused-import
# pylint: disable=g-multiple-import
from tensor2tensor.data_generators.gym_problems import GymDiscreteProblem,\
  GymSimulatedDiscreteProblem, GymRealDiscreteProblem, \
  GymDiscreteProblemWithAutoencoder, GymDiscreteProblemAutoencoded, \
  GymSimulatedDiscreteProblemAutoencoded, \
  GymSimulatedDiscreteProblemForWorldModelEval, \
  GymSimulatedDiscreteProblemForWorldModelEvalAutoencoded
# pylint: enable=g-multiple-import
from tensor2tensor.utils import registry

# Game list from our list of ROMs
# Removed because XDeterministic-v4 did not exist:
# * adventure
# * defender
# * kaboom
ATARI_GAMES = [
    "air_raid", "alien", "amidar", "assault", "asterix", "asteroids",
    "atlantis", "bank_heist", "battle_zone", "beam_rider", "berzerk", "bowling",
    "boxing", "breakout", "carnival", "centipede", "chopper_command",
    "crazy_climber", "demon_attack", "double_dunk", "elevator_action", "enduro",
    "fishing_derby", "freeway", "frostbite", "gopher", "gravitar", "hero",
    "ice_hockey", "jamesbond", "journey_escape", "kangaroo", "krull",
    "kung_fu_master", "montezuma_revenge", "ms_pacman", "name_this_game",
    "phoenix", "pitfall", "pong", "pooyan", "private_eye", "qbert", "riverraid",
    "road_runner", "robotank", "seaquest", "skiing", "solaris",
    "space_invaders", "star_gunner", "tennis", "time_pilot", "tutankham",
    "up_n_down", "venture", "video_pinball", "wizard_of_wor", "yars_revenge",
    "zaxxon"
]

# List from paper:
# https://arxiv.org/pdf/1805.11593.pdf
# plus frostbite.
ATARI_GAMES_WITH_HUMAN_SCORE = [
    "alien", "amidar", "assault", "asterix", "asteroids",
    "atlantis", "bank_heist", "battle_zone", "beam_rider", "bowling",
    "boxing", "breakout", "chopper_command",
    "crazy_climber", "demon_attack", "double_dunk", "enduro",
    "fishing_derby", "freeway", "frostbite", "gopher", "gravitar", "hero",
    "ice_hockey", "jamesbond", "kangaroo", "krull",
    "kung_fu_master", "montezuma_revenge", "ms_pacman", "name_this_game",
    "pitfall", "pong", "private_eye", "qbert", "riverraid",
    "road_runner", "seaquest", "solaris",
    "up_n_down", "video_pinball", "yars_revenge",
]

ATARI_WHITELIST_GAMES = [
    "amidar",
    "bank_heist",
    "berzerk",
    "boxing",
    "crazy_climber",
    "freeway",
    "frostbite",
    "gopher",
    "kung_fu_master",
    "ms_pacman",
    "pong",
    "qbert",
    "seaquest",
]


# Games on which model-free does better than model-based at this point.
ATARI_CURIOUS_GAMES = [
    "bank_heist",
    "boxing",
    "enduro",
    "kangaroo",
    "road_runner",
    "up_n_down",
]


# Games on which based should work.
ATARI_DEBUG_GAMES = [
    "crazy_climber",
    "freeway",
    "pong",
]


# Games for which we hard-define problems to run all around.
# TODO(lukaszkaiser): global registration makes them all rescaled and grayscale,
# no matter the setting of hparams later (as they're registered at start).
ATARI_ALL_MODES_SHORT_LIST = []  # ATARI_DEBUG_GAMES + ATARI_CURIOUS_GAMES


# Different ATARI game modes in OpenAI Gym. Full list here:
# https://github.com/openai/gym/blob/master/gym/envs/__init__.py
ATARI_GAME_MODES = [
    "Deterministic-v0",  # 0.25 repeat action probability, 4 frame skip.
    "Deterministic-v4",  # 0.00 repeat action probability, 4 frame skip.
    "NoFrameskip-v0",    # 0.25 repeat action probability, 1 frame skip.
    "NoFrameskip-v4",    # 0.00 repeat action probability, 1 frame skip.
    "-v0",               # 0.25 repeat action probability, (2 to 5) frame skip.
    "-v4"                # 0.00 repeat action probability, (2 to 5) frame skip.
]

# List of all ATARI envs in all modes.
ATARI_PROBLEMS = {}


@registry.register_problem
class GymWrappedFullPongRandom(GymDiscreteProblem):
  """Pong game, random actions."""

  @property
  def env_name(self):
    return "T2TPongWarmUp20RewSkipFull-v1"

  @property
  def min_reward(self):
    return -1

  @property
  def num_rewards(self):
    return 3

  @property
  def num_testing_steps(self):
    return 100


@registry.register_problem
class GymDiscreteProblemWithAgentOnWrappedFullPong(GymRealDiscreteProblem,
                                                   GymWrappedFullPongRandom):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnWrappedFullPongWithAutoencoder(
    GymDiscreteProblemWithAutoencoder, GymWrappedFullPongRandom):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnWrappedFullPongAutoencoded(
    GymDiscreteProblemAutoencoded, GymWrappedFullPongRandom):
  pass


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgentOnWrappedFullPong(
    GymSimulatedDiscreteProblem, GymWrappedFullPongRandom):
  """Simulated pong."""

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_wrapped_full_pong"

  @property
  def num_testing_steps(self):
    return 100


@registry.register_problem
class GymSimulatedDiscreteProblemForWorldModelEvalWithAgentOnWrappedFullPong(
    GymSimulatedDiscreteProblemForWorldModelEval, GymWrappedFullPongRandom):
  """Simulated pong for world model evaluation."""

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_wrapped_full_pong"

  @property
  def num_testing_steps(self):
    return 100


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgentOnWrappedFullPongAutoencoded(
    GymSimulatedDiscreteProblemAutoencoded, GymWrappedFullPongRandom):
  """GymSimulatedDiscreteProblemWithAgentOnWrappedFullPongAutoencoded."""

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_wrapped_full_pong_autoencoded"

  @property
  def num_testing_steps(self):
    return 100


@registry.register_problem
class GymSimulatedDiscreteProblemForWorldModelEvalWithAgentOnWrappedFullPongAutoencoded(  # pylint: disable=line-too-long
    GymSimulatedDiscreteProblemForWorldModelEvalAutoencoded,
    GymWrappedFullPongRandom):
  """Simulated pong for world model evaluation with encoded frames."""

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_wrapped_full_pong_autoencoded"

  @property
  def num_testing_steps(self):
    return 100


class GymClippedRewardRandom(GymDiscreteProblem):
  """Abstract base class for clipped reward games."""

  @property
  def env_name(self):
    raise NotImplementedError

  @property
  def min_reward(self):
    return -1

  @property
  def num_rewards(self):
    return 3


def create_problems_for_game(
    game_name,
    resize_height_factor=2,
    resize_width_factor=2,
    grayscale=True,
    game_mode="Deterministic-v4",
    autoencoder_hparams=None):
  """Create and register problems for game_name.

  Args:
    game_name: str, one of the games in ATARI_GAMES, e.g. "bank_heist".
    resize_height_factor: factor by which to resize the height of frames.
    resize_width_factor: factor by which to resize the width of frames.
    grayscale: whether to make frames grayscale.
    game_mode: the frame skip and sticky keys config.
    autoencoder_hparams: the hparams for the autoencoder.

  Returns:
    dict of problems with keys ("base", "agent", "simulated").

  Raises:
    ValueError: if clipped_reward=False or game_name not in ATARI_GAMES.
  """
  if game_name not in ATARI_GAMES:
    raise ValueError("Game %s not in ATARI_GAMES" % game_name)
  if game_mode not in ATARI_GAME_MODES:
    raise ValueError("Unknown ATARI game mode: %s." % game_mode)
  camel_game_name = "".join(
      [w[0].upper() + w[1:] for w in game_name.split("_")])
  camel_game_name += game_mode
  env_name = camel_game_name

  # Create and register the Random and WithAgent Problem classes
  problem_cls = type("Gym%sRandom" % camel_game_name,
                     (GymClippedRewardRandom,),
                     {"env_name": env_name,
                      "resize_height_factor": resize_height_factor,
                      "resize_width_factor": resize_width_factor,
                      "grayscale": grayscale})
  registry.register_problem(problem_cls)

  with_agent_cls = type("GymDiscreteProblemWithAgentOn%s" % camel_game_name,
                        (GymRealDiscreteProblem, problem_cls), {})
  registry.register_problem(with_agent_cls)

  with_ae_cls = type(
      "GymDiscreteProblemWithAgentOn%sWithAutoencoder" % camel_game_name,
      (GymDiscreteProblemWithAutoencoder, problem_cls),
      {"ae_hparams_set": autoencoder_hparams})
  registry.register_problem(with_ae_cls)

  ae_cls = type(
      "GymDiscreteProblemWithAgentOn%sAutoencoded" % camel_game_name,
      (GymDiscreteProblemAutoencoded, problem_cls),
      {"ae_hparams_set": autoencoder_hparams})
  registry.register_problem(ae_cls)

  # Create and register the simulated Problem
  simulated_cls = type(
      "GymSimulatedDiscreteProblemWithAgentOn%s" % camel_game_name,
      (GymSimulatedDiscreteProblem, problem_cls), {
          "initial_frames_problem": with_agent_cls.name,
          "num_testing_steps": 100
      })
  registry.register_problem(simulated_cls)

  simulated_ae_cls = type(
      "GymSimulatedDiscreteProblemWithAgentOn%sAutoencoded" % camel_game_name,
      (GymSimulatedDiscreteProblemAutoencoded, problem_cls), {
          "initial_frames_problem": ae_cls.name,
          "num_testing_steps": 100,
          "ae_hparams_set": autoencoder_hparams
      })
  registry.register_problem(simulated_ae_cls)

  # Create and register the simulated Problem
  world_model_eval_cls = type(
      "GymSimulatedDiscreteProblemForWorldModelEvalWithAgentOn%s" %
      camel_game_name,
      (GymSimulatedDiscreteProblemForWorldModelEval, problem_cls), {
          "initial_frames_problem": with_agent_cls.name,
          "num_testing_steps": 100,
          "ae_hparams_set": autoencoder_hparams
      })
  registry.register_problem(world_model_eval_cls)

  world_model_eval_ae_cls = type(
      "GymSimulatedDiscreteProblemForWorldModelEvalWithAgentOn%sAutoencoded" %
      camel_game_name,
      (GymSimulatedDiscreteProblemForWorldModelEvalAutoencoded, problem_cls), {
          "initial_frames_problem": ae_cls.name,
          "num_testing_steps": 100,
          "ae_hparams_set": autoencoder_hparams
      })
  registry.register_problem(world_model_eval_ae_cls)


# Register the atari games with all of the possible modes.
for game in ATARI_ALL_MODES_SHORT_LIST:
  for mode in ATARI_GAME_MODES:
    create_problems_for_game(game, game_mode=mode)
