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

# ZONE_NAMES = ['us-central1-c','us-east1-c','us-east1-d','europe-west1-b','europe-west1-d','asia-east1-a','asia-east1-b','us-west1-b','us-central1-a']
# HAPARMS = ['resnet_cifar_32','resnet_50']

def run(job_name, num_worker, max_worker, model, hparam_set, problem, train_steps, ckpt_frequency, automation_test='0', run='0', profile='0'):
    job = resource_acquisition.ResourceManager()
    # job.create_instance(COMPUTE, PROJECTNAME, zone, name, False, '4', True)
    job.acquire_resource(job_name+'-'+num_worker+'-'+max_worker, 1, 4, int(num_worker), 100)

    if num_worker == max_worker:
        set_slot = '0'
    else:
        set_slot = '1'

    worker_temp = str(int(num_worker) - 1)

    subprocess.call(
        ["./start_one_time_training.sh", job_name, '1', worker_temp, '1', 'gs://shijian-18-ml', model, hparam_set,
         problem, train_steps, ckpt_frequency, '0', '0', '0', set_slot, max_worker])

    # os.system("gcloud compute scp --zone " + zone + " ozymandias@" + name + ":~/ping.txt ${HOME}/desktop/spotTrain_data/1_29/"+zone+"_"+hparam_set+"_ping.txt")
    # os.system("gcloud compute scp --zone " + zone + " ozymandias@" + name + ":~/iperf3.txt ${HOME}/desktop/spotTrain_data/1_29/" + zone + "_" + hparam_set + "_iperf3.txt")
    if int(num_worker) + 1 < 3:
        os.system(
                "gcloud compute instances delete --zone us-east1-c " + job_name+'-'+num_worker+'-'+max_worker + "-ps-0 -q &")
        os.system(
                "gcloud compute instances delete --zone us-east1-c " + job_name+'-'+num_worker+'-'+max_worker + "-master -q &")
        for i in range(4):
            os.system(
                "gcloud compute instances delete --zone us-east1-c " + job_name+'-'+num_worker+'-'+max_worker + "-worker-"+str(i) + " -q &")
    else:
        os.system(
            "gcloud compute instances delete --zone us-east1-c " + job_name+'-'+num_worker+'-'+max_worker + "-ps-0 -q &")
        os.system(
            "gcloud compute instances delete --zone us-east1-c " + job_name+'-'+num_worker+'-'+max_worker + "-master -q &")
        for i in range(4):
            os.system(
                "gcloud compute instances delete --zone us-east1-c " + job_name+'-'+num_worker+'-'+max_worker + "-worker-" + str(i) + " -q &")
        time.sleep(300)

def main(case):
    hparam = "resnet_cifar_32_vanilla"
    step = 16000
    if case == '1':
        for i in range(1, 5):
            run('a-lr-case1', str(i), '19', 'resnet', hparam, 'image_cifar10', str(step * i), '4000')
    else:
        for j in range(1, 5):
            run('a-lr-case2', str(j), str(j), 'resnet', hparam, 'image_cifar10', str(step * j), '4000')

    # step = 120000
    # run('a-lr-baseline', '1', '1', 'resnet', hparam, 'image_cifar10', str(step * 1), '6000')

# main('us-west1-b','resnet','resnet_cifar_15','image_cifar10','390','100000')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--case',
        type=str,
        required=True,
        help='Which test case to use')
    args = parser.parse_args()

    main(**vars(args))
    # main(1)