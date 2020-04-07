from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
  
import sys
import argparse
import math
import datetime
import os
import json
import xmlrpclib
import socket
import googleapiclient
import googleapiclient.discovery

import tensorflow as tf
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training_util
from tensorflow.python.training import session_run_hook
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.core.util.event_pb2 import SessionLog

name = socket.gethostname()
compute = googleapiclient.discovery.build('compute', 'v1')
projectName = "shijian-18"
rpc_server_name = name.split('-')[0] + '-' + 'ps-0'
request = compute.instances().get(project=projectName, zone='us-west1-b', instance=rpc_server_name)
response = request.execute()
ps_ip = response['networkInterfaces'][0]['networkIP'].encode("utf-8")
xml_server = xmlrpclib.ServerProxy('http://'+ ps_ip +':8000')
tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
task_type = tf_config['task']['type'].encode("utf-8")
task_index = str(tf_config['task']['index'])
is_training = 0

class custom_step_hook(basic_session_run_hooks.CheckpointSaverHook):

    def __init__(self,
               every_n_steps=None,
               every_n_secs=2,
               output_dir=None,
               summary_writer=None):

        if (every_n_steps is None) == (every_n_secs is None):
          raise ValueError(
              "exactly one of every_n_steps and every_n_secs should be provided.")
        self._timer = basic_session_run_hooks.SecondOrStepTimer(
            every_steps=every_n_steps, every_secs=every_n_secs)

        # self._summary_writer = summary_writer
        self._output_dir = output_dir
        self._last_global_step = None
        self._steps_per_run = 1

    def begin(self):
        # if self._summary_writer is None and self._output_dir:
        #   self._summary_writer = SummaryWriterCache.get(self._output_dir)
        self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
        if self._global_step_tensor is None:
          raise RuntimeError(
              "Global step should be created to use StepCounterHook.")
        self._summary_tag = training_util.get_global_step().op.name + "/sec"

    def after_create_session(self, session, coord):
        global_step = session.run(self._global_step_tensor)
        # We do write graph and saver_def at the first call of before_run.
        # We cannot do this in begin, since we let other hooks to change graph and
        # add variables in begin. Graph is finalized after all begin calls.
        # training_util.write_graph(
        #     ops.get_default_graph().as_graph_def(add_shapes=True),
        #     self._checkpoint_dir,
        #     "graph.pbtxt")
        # saver_def = self._get_saver().saver_def if self._get_saver() else None
        # graph = ops.get_default_graph()
        # meta_graph_def = meta_graph.create_meta_graph_def(
        #     graph_def=graph.as_graph_def(add_shapes=True),
        #     saver_def=saver_def)
        # self._summary_writer.add_graph(graph)
        # self._summary_writer.add_meta_graph(meta_graph_def)
        # # The checkpoint saved here is the state at step "global_step".
        # self._save(session, global_step)
        self._timer.update_last_triggered_step(global_step)

    def before_run(self, run_context):  # pylint: disable=unused-argument
        global task_type
        global task_index
        global name
        global is_training
        if is_training == 0:
            xml_server.serverStarted(task_type, task_index, name)
            is_training = 1
        return SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        global task_type
        global task_index
        _ = run_context

        stale_global_step = run_values.results
        if self._timer.should_trigger_for_step(
            stale_global_step + self._steps_per_run):
            # get the real value after train op.
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
                global_step)
                current_master = (xml_server.getMaster()[0], xml_server.getMaster()[1])
                if current_master == (task_type, task_index):
                    if elapsed_time is not None and xml_server.getDisplaySignal() != 0:
                        xml_server.displaySteps(elapsed_steps.item(), elapsed_time, global_step.item())
                        # self._log_and_record(elapsed_steps, elapsed_time, global_step)

            if stale_global_step == self._last_global_step:
                logging.log_first_n(
                    logging.WARN,
                    "It seems that global step (tf.train.get_global_step) has not "
                    "been increased. Current value (could be stable): %s vs previous "
                    "value: %s. You could increase the global step by passing "
                    "tf.train.get_global_step() to Optimizer.apply_gradients or "
                    "Optimizer.minimize.", 5, stale_global_step, self._last_global_step)

        self._last_global_step = stale_global_step