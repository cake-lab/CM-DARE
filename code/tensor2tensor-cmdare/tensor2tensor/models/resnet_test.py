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

"""Resnet tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.layers import modalities
from tensor2tensor.models import resnet
from tensor2tensor.models import shake_shake
from tensor2tensor.utils import optimize

import tensorflow as tf
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training_util
from tensorflow.python.training import session_run_hook
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.core.util.event_pb2 import SessionLog

def resnet_tiny_cpu():
  hparams = resnet.resnet_base()
  hparams.layer_sizes = [2, 2, 2, 2]
  hparams.use_nchw = False
  return hparams

def resnet_50():
  hparams = resnet.resnet_50()
  hparams.use_nchw = False
  return hparams

def resnet_32():
  #hparams = resnet.resnet_custom8()
  hparams = shake_shake.shakeshake_big()
  hparams.use_nchw = False
  return hparams

class ResnetTest(tf.test.TestCase):

  def _test_resnet(self, img_size, output_size):
    vocab_size = 1
    batch_size = 1
    x = np.random.random_integers(
        0, high=255, size=(batch_size, img_size, img_size, 3))
    y = np.random.random_integers(
        1, high=vocab_size, size=(batch_size, 1, 1, 1))
    #hparams = resnet_tiny_cpu()
    #hparams = resnet_50()
    hparams = resnet_32()
    p_hparams = problem_hparams.test_problem_hparams(vocab_size,
                                                     vocab_size,
                                                     hparams)
    p_hparams.input_modality["inputs"] = modalities.ImageModality(hparams)
    p_hparams.target_modality = modalities.ClassLabelModality(
        hparams, vocab_size)
    run_meta = tf.RunMetadata()
    with self.test_session() as session:
      features = {
          "inputs": tf.constant(x, dtype=tf.int32),
          "targets": tf.constant(y, dtype=tf.int32),
      }
      #model = resnet.Resnet(hparams, tf.estimator.ModeKeys.TRAIN, p_hparams)
      model = shake_shake.ShakeShake(hparams, tf.estimator.ModeKeys.TRAIN, p_hparams)
      logits, _ = model(features)
      print(logits.get_shape())
      #opts = tf.profiler.ProfileOptionBuilder.float_operation()
      #flops = tf.profiler.profile(tf.get_default_graph(), run_meta=run_meta, options=opts)
      #print(flops.total_float_ops)
      session.run(tf.global_variables_initializer())
      #res = session.run(logits)
      tf.get_variable_scope().set_initializer(
        optimize.get_variable_initializer(hparams))
      loss = tf.losses.sparse_softmax_cross_entropy(
            labels=tf.constant(0, dtype=tf.int32, shape=[1,1,1,1,1]),
            logits=logits)
      train_op = optimize.optimize(loss, 0.1, hparams)
      session.run(loss)
      opts = tf.profiler.ProfileOptionBuilder.float_operation()
      flops = tf.profiler.profile(tf.get_default_graph(), run_meta=run_meta, options=opts)
      print(flops.total_float_ops)
    #self.assertEqual(res.shape, (batch_size,) + output_size + (1, vocab_size))

  def testResnetLarge(self):
    self._test_resnet(img_size=32, output_size=(1, 1))

if __name__ == "__main__":
  tf.test.main()
