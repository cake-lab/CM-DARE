import os
import argparse
import sys
import subprocess
import time
import googleapiclient.discovery
import googleapiclient
import resource_acquisition
import threading
import paramiko

PROJECTNAME = "shijian-18"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/Users/ozymandias/Desktop/cloudComputing/shijian-18-key.json'
COMPUTE = googleapiclient.discovery.build('compute', 'v1')

ZONE_NAMES = ['us-central1-c','us-east1-c','us-east1-d','europe-west1-b','europe-west1-d','asia-east1-a','asia-east1-b','us-west1-b','us-central1-a']
HAPARMS = ['resnet_cifar_32','resnet_50']

def main(zone, model, hparam_set, problem, train_steps, ckpt_frequency, automation_test='0', run='0', profile='0'):
    temp = ''
    for c in hparam_set:
        if c == "_":
            temp += '-'
        else:
            temp += c
    name = zone + '-' + temp
    job = resource_acquisition.ResourceManager()
    job.create_instance(COMPUTE, PROJECTNAME, zone, name, False, '4', True)
    request = COMPUTE.instances().get(project=PROJECTNAME, zone=zone, instance=name)
    result = request.execute()
    command = "echo VM READY"
    username = "ozymandias"
    home_var = os.environ['HOME']
    while result['status'] != 'RUNNING':
        request = COMPUTE.instances().get(project=PROJECTNAME, zone=zone, instance=name)
        result = request.execute()
    ip = result['networkInterfaces'][0]['accessConfigs'][0]['natIP']
    port = 22
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.WarningPolicy)
    while True:
        try:
            client.connect(ip, port=port, username=username, password=None,
                           key_filename=home_var + "/.ssh/google_compute_engine")

            stdin, stdout, stderr = client.exec_command(command)
            break
        except (paramiko.ssh_exception.BadHostKeyException, paramiko.ssh_exception.AuthenticationException,
                paramiko.ssh_exception.SSHException, paramiko.ssh_exception.socket.error) as e:
            print "Retrying SSH to VM"
            time.sleep(1)
    client.close()

    subprocess.call(
        ["./start_one_time_training.sh", name, '1', '0', '1', 'gs://shijian-18-ml', model, hparam_set,
         problem, train_steps, ckpt_frequency, '0', '0', '0'])

    os.system("gcloud compute scp --zone " + zone + " ozymandias@" + name + ":~/ping.txt ${HOME}/desktop/spotTrain_data/1_29/"+zone+"_"+hparam_set+"_ping.txt")
    os.system("gcloud compute scp --zone " + zone + " ozymandias@" + name + ":~/iperf3.txt ${HOME}/desktop/spotTrain_data/1_29/" + zone + "_" + hparam_set + "_iperf3.txt")

    os.system("gcloud compute instances delete --zone " + zone + " " + name + " -q &")

# main('us-west1-b','resnet','resnet_cifar_15','image_cifar10','390','100000')
if __name__ == '__main__':
    for zone in ZONE_NAMES:
        for hparam in HAPARMS:
            main(zone,'resnet',hparam,'image_cifar10','500','100000')