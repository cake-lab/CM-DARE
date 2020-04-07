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
from tensorflow.core.util.event_pb2 import SessionLog

name = socket.gethostname()
compute = googleapiclient.discovery.build('compute', 'v1')
projectName = "shijian-18"
rpc_server_name = name.split('-')[0] + '-' + 'ps-0'
try:
    request = compute.instances().get(project=projectName, zone='us-west1-b', instance=rpc_server_name)
    response = request.execute()
    ps_ip = response['networkInterfaces'][0]['networkIP'].encode("utf-8")
    xml_server = xmlrpclib.ServerProxy('http://'+ ps_ip +':8000')
except:
    print("Cluster not properly set yet")

class custom_saver_hook(basic_session_run_hooks.CheckpointSaverHook):
    def _save(self, session, step):
        """Saves the latest checkpoint, returns should_stop."""
        global xml_server
        tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
        task_type = tf_config['task']['type'].encode("utf-8")
        task_index = str(tf_config['task']['index'])
        current_master = (xml_server.getMaster()[0], xml_server.getMaster()[1])
        if current_master == (task_type, task_index):
            logging.info("Saving checkpoints for %d into %s.", step, self._save_path)
            logging.info("Current checkpoint is dumped by %s %s", task_type, task_index)

            for l in self._listeners:
              l.before_save(session, step)

            self._get_saver().save(session, self._save_path, global_step=step)
            self._summary_writer.add_session_log(
                SessionLog(
                    status=SessionLog.CHECKPOINT, checkpoint_path=self._save_path),
                step)

            should_stop = False
            for l in self._listeners:
              if l.after_save(session, step):
                logging.info(
                    "A CheckpointSaverListener requested that training be stopped. "
                    "listener: {}".format(l))
                should_stop = True
            return should_stop