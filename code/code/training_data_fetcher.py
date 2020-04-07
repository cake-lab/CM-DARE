import os
import argparse
import sys
import subprocess
import time
import googleapiclient.discovery
import googleapiclient
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import csv
import numpy as np

def main(runs, setting, target_dir):
    with open(target_dir+'/train_data.csv', mode='w') as input:
        writer = csv.writer(input, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # writer.writerow(['Wall Time', 'Step', 'Accuracy', 'Zone', 'Run'])
        writer.writerow(['Wall Time', 'Step', 'Accuracy', 'Run'])
        # for zone in zones:
        #     # if zone == 'us-west1-b':
        #     #     runs = 26
        #     # elif zone == 'us-central1-a':
        #     #     runs = 17
        #     # else:
        #     #     runs = 2
        #     for run in range(1,runs+1):
        #         # if run == 23:
        #         #     continue
        #         w_times, step_nums, vals = ckpt_fetch(None, None, zone, run)
        #         for i in range(len(vals)):
        #             writer.writerow([w_times[i], step_nums[i], vals[i], zone, run])
        for run in range(1,runs+1):
            w_times, step_nums, vals = ckpt_fetch(None, None, None, run, setting)
            for i in range(len(vals)):
                writer.writerow([w_times[i], step_nums[i], vals[i], run])


def ckpt_fetch(latency, model, zone, run, setting=None):
    # event_acc = EventAccumulator('gs://shijian-18-ml/30-cluster/k80-demand-'+zone+'-run'+str(run)+'/eval')
    event_acc = EventAccumulator('gs://shijian-18-ml/30-cluster/a-' + setting + '-run' + str(run) + '/eval')
    event_acc.Reload()
    # Show all tags in the log file
    # print(event_acc.Tags())
    # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
    # w_times, step_nums, vals = zip(*event_acc.Scalars('global_step/sec'))
    w_times, step_nums, vals = zip(*event_acc.Scalars('metrics-image_cifar10/targets/accuracy'))
    return w_times, step_nums, vals

def latency_fetch(latency, model, run):
    file = "latency"+ latency +"ms_resnet_cifar_"+ model +"_run" + run
    os.system(
        "gcloud compute scp ozymandias@master-0:~/latency"+ latency +"ms_resnet_cifar_"+ model +"_run" + run + ".txt" + " ~/desktop/spotTrain_data/2_6")

def latency_read(latency, model, run):
    file = "latency" + latency + "ms_resnet_cifar_" + model + "_run" + run
    val = os.popen("tail -n 1 " + '~/desktop/spotTrain_data/2_6/' + file + ".txt | cut -d'/' -f5").read()
    return val

# def main():


if __name__ == '__main__':
    models = ['15','32_vanilla']
    # zones = ['us-east1-c','us-central1-c','us-west1-b','europe-west1-b']
    zones= ['us-east1-c']
    runs = 5
    # for i in range(2,9):
    #     main(i)
    # main(zones, runs)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--runs',
        type=int,
        required=True,
        help='Pff')
    parser.add_argument(
        '--setting',
        type=str,
        required=True,
        help='Pff')
    parser.add_argument(
        '--target_dir',
        type=str,
        required=True,
        help='Pff')
    args = parser.parse_args()

    main(**vars(args))

    # main('zoo', runs, '2east-1central-1west')

    # for model in models:
    #     if model == '15':
    #         run_num = 6
    #     else:
    #         run_num = 3
    #     for latency in range(0, 210, 10):
    #         for run in range(1,run_num):
    #             latency_fetch(str(latency), model, str(run))

    ### Run this on cloud VM
    # with open('worker_data.csv', mode='w') as input:
    #     writer = csv.writer(input, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow(['Latency', 'Model', 'Run', 'Speed'])
    #     for model in models:
    #         if model == '15':
    #             run_num = 6
    #         else:
    #             run_num = 3
    #         for latency in range(0, 210, 10):
    #             for run in range(1,run_num):
    #                 val = ckpt_fetch(str(latency), model, str(run))
    #                 writer.writerow([str(latency), model, str(run), np.mean(val[1:])])


    # with open('/Users/ozymandias/desktop/spotTrain_data/2_6/latency.csv', mode = 'w') as input:
    #     writer = csv.writer(input, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow(['Latency','Model','Run'])
    #     for model in models:
    #         if model == '15':
    #             run_num = 6
    #         else:
    #             run_num = 3
    #         for latency in range(0, 210, 10):
    #             for run in range(1,run_num):
    #                 val = latency_read(str(latency), model, str(run))
    #                 writer.writerow([val, model, str(run)])

