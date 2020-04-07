from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler
import time
from datetime import datetime
import googleapiclient.discovery
import googleapiclient
import subprocess
import os
import socket
import resource_acquisition
import threading
from datastore import StatusDAO as dao

# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

# Create RPC server
compute = googleapiclient.discovery.build('compute', 'v1')
projectName = "shijian-18"
request = compute.instances().get(project=projectName, zone='us-west1-b', instance=socket.gethostname())
response = request.execute()
ip = response['networkInterfaces'][0]['networkIP'].encode("utf-8")
server = SimpleXMLRPCServer((ip, 8000),
                            requestHandler=RequestHandler, logRequests=False)

jobName = socket.gethostname().split('-')[0]
max_worker_num = 100
current_master = ("chief", "0")
running_k80 = 4 ## TODO: should be able to give initial value
running_p100 = 0
running_v100 = 0
running_workers = set([])
worker_array = [0] * max_worker_num
downed_server = 0
display_signal = 0
print "RPC server started"
# substitute_number = 0
# db = dao('/home/ozymandias/trans_recovery.db')
## Need a hashtable here to record worker status, so that we can determine who's the next chief node
## Maybe hashtable is too costly, if we keep a list(or set) of currently running workers it could be easier

## Handling the signal that a worker has started working
def serverStarted(server_type, server_index, server_name):
	print("Server starts training at: " + str(datetime.utcnow()))
	global running_workers
	global worker_array
	global downed_server
	global display_signal
	server_type = server_type.encode("utf-8")
	server_index = str(server_index)
	tup = tuple([server_type, server_index])
	running_workers.add(tup)
	worker_array[int(server_index)] = 1
	if downed_server != 0:
		display_signal = 2
		downed_server -= 1
	# db.add_entry(server_name, creation_time=str(datetime.utcnow()))
	return True
server.register_function(serverStarted)

def getWorkers():
	global running_workers
	return list(running_workers)
server.register_function(getWorkers)

def getDisplaySignal():
	global display_signal
	return display_signal
server.register_function(getDisplaySignal)

def displaySteps(elapsed_steps, elapsed_time, global_step):
	global display_signal
	global downed_server
	f = open("/home/ozymandias/train_log.txt", "a")
	f.write(str(datetime.utcnow()) + "\n")
	if display_signal == 1:
		f.write("A server is revoked, logging steps information\n")
	elif display_signal == 2:
		f.write("A server is started, logging steps information\n")
	f.write("Current global step: " + str(global_step) + "\n")
	f.write("Elased steps: " + str(elapsed_steps) + "\n")
	f.write("Elased time: " + str(elapsed_time) + "\n")
	if display_signal == 2:
		f.write("\n")
	f.close()
	display_signal = 0
	return True
server.register_function(displaySteps)

## Handling the shutdown signal passed from the workers
def serverDown(server_type, server_index, server_name):
	print("Received server down signal at: " + str(datetime.utcnow()))
	global running_workers
	# global substitute_number
	global worker_array
	global max_worker_num
	global current_master
	global jobName
	global downed_server
	global display_signal
	downed_server += 1
	display_signal = 1
	server_type = server_type.encode("utf-8")
	server_index = str(server_index)
	tup = tuple([server_type, server_index])
	## If the server if current chief then switch it to next worker
	if tup == current_master:
		current_master = next(iter(running_workers))
	# for i in running_workers:
	# 	print("Printing running workers", i)
	if tup in running_workers:
		print("Worker is down: ", tup)
		running_workers.remove(tup)
	else:
		print("This worker is not on the running list")

	# predict(8000, 2000, 70, 19.4, 4.7)

	## Get a substitute. Shutting down is way too fast, the thread closes off before acquisition happens, need to address that.
	for i in range(max_worker_num):
		if tuple(['worker', str(i)]) not in running_workers and worker_array[i] != 1:
			vmName = jobName + '-' + 'worker' + '-' + str(i)
			# startSubJob(jobName, vmName, str(i))
			thread = startSubJob(jobName, vmName, str(i))
			thread.start()
			break

	## For testing purpose
	# if server_name != 'instance-4':
	# 	check_delete_process("us-central1-a", server_name)

	## Database related
	# try:
	# 	db.update_termination_time(server_name, str(datetime.utcnow()))
	# except:
	# 	db.add_entry(server_name, termination_time=str(datetime.utcnow()))

	# if 'sub' in server_name.split('-'):
	# 	return True

	## Not sure we need this to get substitute
	# substitute_number += 1
	# sub_name = server_name + "-sub-" + str(substitute_number)
	# startNewInstance(name, "us-central1-a", "k80", True, True) ## Not really needed
	# subprocess.Popen(["parallel","python","~/proj_code/code/create.py",":::","--name="+sub_name+"-in-k80", "--name="+sub_name+"-in-p100", "--name="+sub_name+"-in-v100", "--name="+sub_name+"-cross-k80", ":::+", "--gpu=k80", "--gpu=p100", "--gpu=v100", "--gpu=k80", ":::+", "--zone=us-west1-b", "--zone=us-west1-b", "--zone=us-west1-b", "--zone=us-east1-c"],stdout=subprocess.PIPE)
	# check_instance_status("us-central1-a", name)

	return True
server.register_function(serverDown)

class startSubJob(threading.Thread):
	def __init__(self, jobname, vmname, index):
		threading.Thread.__init__(self)
		self.compute = googleapiclient.discovery.build('compute', 'v1')
		self.projectName = "shijian-18"
		self.index = index
		self.vmname = vmname
		self.jobName = jobname

	def run(self):
		manager = resource_acquisition.ResourceManager()
		startNewInstance(self.vmname, 'us-west1-b', 'k80', True, True) ## TODO: Zone and GPU needs to be automated
		int_ip, ip = manager.check_instance_status(self.compute, self.projectName, 'us-west1-b', self.vmname)
		subprocess.call(["/home/ozymandias/proj_code/code/spotTrain/start_sub.sh", self.jobName, self.vmname, str(self.index), 'us-west1-b'])
		return

# def startSubJob(jobname, vmname, index):
# 	manager = resource_acquisition.ResourceManager()
# 	startNewInstance(vmname, 'us-west1-b', 'k80', True, True) ## TODO: Zone and GPU needs to be automated
# 	int_ip, ip = manager.check_instance_status(compute, projectName, 'us-west1-b', vmname)
# 	subprocess.call(["./start_sub.sh"], jobName, vmname, str(i), 'us-west1-b')
# 	return

## TODO: Predictive model
def predict(total_steps, current_steps, request_overhead, current_speed, new_server_speed):
	estimated_time = request_overhead + (total_steps - current_steps) / new_server_speed - request_overhead * (current_speed / new_server_speed)
	return estimated_time

def getNumberOfActiveWorkers():
	return len(running_workers)

def getMaster():
	global current_master
	return current_master
server.register_function(getMaster)

def setMaster(master_type, master_index):
    global current_master
    if current_master != None:
    	current_master = (master_type, str(master_index))
    print "Master node has changed to:" + str(current_master)
    return current_master
server.register_function(setMaster)

## Create a new instance to replace the revoked instance after receiving the shutdown signal
def startNewInstance(name, zone, gpu_type, startup, shutdown):
	print("Starting to request new instance of type " + gpu_type + " in zone " + zone + " at: " + str(datetime.utcnow()))
	cpu_num = 4
	is_gpu = True
	preemtible = False
	compute = googleapiclient.discovery.build('compute', 'v1')
	project = "shijian-18"
	# while True:
	# 	try:
	if is_gpu:
		# other images to use: gpu-custom-tf
		image_response = compute.images().get(project=project, image='gpu-ubuntu18-2').execute()
		# image_response = compute.images().get(project=PROJECTNAME, image='gpu-custom-tf').execute()
		source_disk_image = image_response['selfLink']
		if gpu_type == 'k80':
			gpu = 'nvidia-tesla-k80'
			machine_type = "zones/%s/machineTypes/custom-4-51200-ext" % (zone)
		elif gpu_type == 'v100':
			gpu = 'nvidia-tesla-v100'
			machine_type = "zones/%s/machineTypes/custom-8-62464-ext" % (zone)
		elif gpu_type == 'p100':
			gpu = 'nvidia-tesla-p100'
			machine_type = "zones/%s/machineTypes/custom-8-62464-ext" % (zone)
	else:
		image_response = compute.images().get(project=project, image='cpu-updated').execute()
		source_disk_image = image_response['selfLink']
		machine_type = "zones/%s/machineTypes/custom-%s-16384-ext" % (zone, str(cpu_num))

	if startup:
		startup_script = open(
			os.path.join(
			os.path.dirname(__file__), 'startup.py'), 'r').read()
	else:
		startup_script = None

	if shutdown:
		shutdown_script = open(
			os.path.join(
			os.path.dirname(__file__), 'shutdown.py'), 'r').read()
	else:
		shutdown_script = None

	if is_gpu:
		config = {
			'name': name,
			'machineType': machine_type,
			"minCpuPlatform": "Intel Broadwell", #CPU Platform spec
			'scheduling': [
				{
					'preemptible': preemtible,
					"onHostMaintenance": "terminate",
				}
			],
			'serviceAccounts': [
				{
					"email": "[SERVICE_ACCOUNT_EMAIL]",
					"scopes": ["https://www.googleapis.com/auth/cloud-platform"]
				}
			],

			'disks': [
				{
					'boot': True,
					'autoDelete': True,
					'diskSizeGb': '100',
					'initializeParams': {
						'sourceImage': source_disk_image,
					}
				}
			],

			'networkInterfaces': [{
				'network': 'global/networks/default',
				# 'networkIP': '10.142.0.24',
				'accessConfigs': [
					{'type': 'ONE_TO_ONE_NAT', 'name': 'External NAT'}
				]
			}],

			'serviceAccounts': [{
				'email': 'default',
				'scopes': [
					'https://www.googleapis.com/auth/devstorage.read_write',
					'https://www.googleapis.com/auth/logging.write',
					'https://www.googleapis.com/auth/cloud-platform'
				]
			}],

			"guestAccelerators":
				[
					{
						"acceleratorCount": 1,
						"acceleratorType": "https://www.googleapis.com/compute/v1/projects/" + project + "/zones/" + zone + "/acceleratorTypes/" + gpu
					}
				],

			'metadata': {
				'items': [
					{
						'key': 'startup-script',
						'value': startup_script
					},
					{
						'key': 'shutdown-script',
						'value': shutdown_script
					}
				]
			}
		}
	else:
		config = {
			'name': name,
			'machineType': machine_type,
			"minCpuPlatform": "Intel Broadwell",
			'scheduling': [
				{
					'preemptible': preemtible,
					"onHostMaintenance": "terminate",
				}
			],
			'serviceAccounts': [
				{
					 "email": "[SERVICE_ACCOUNT_EMAIL]",
					"scopes": ["https://www.googleapis.com/auth/cloud-platform"]
				}
			],

			'disks': [
				{
					'boot': True,
					'autoDelete': True,
					'diskSizeGb': '100',
					'initializeParams': {
						'sourceImage': source_disk_image,
					}
				}
			],

            'networkInterfaces': [{
				'network': 'global/networks/default',
				# 'networkIP': '10.142.0.24',
				'accessConfigs': [
					{'type': 'ONE_TO_ONE_NAT', 'name': 'External NAT'}
				]
			}],

			'serviceAccounts': [{
				'email': 'default',
				'scopes': [
					'https://www.googleapis.com/auth/devstorage.read_write',
					'https://www.googleapis.com/auth/logging.write',
					'https://www.googleapis.com/auth/cloud-platform'
				]
			}]
		}
	compute.instances().insert(
		project=project,
		zone=zone,
		body=config).execute()
		# except googleapiclient.errors.HttpError as e:
		# 	print("Error occured when creating instance") + e
		# 	pass
		# except:
		# 	print("Error!")
	return True

def check_instance_status(zone, name):
	compute = googleapiclient.discovery.build('compute', 'v1')
	project = "shijian-18"
	then = time.time() * 1000
	flag = 1
	# print 'Initial timestamp: ', then
	while True:
		request = compute.instances().get(project=project, zone=zone, instance=name)
		result = request.execute()

		if result['status'] == 'STAGING' and flag == 1:
			now2 = time.time() * 1000
			# print "Staging timestamp: ", now2
			if 'error' in result:
			    raise Exception(result['error'])
			provisioning = now2 - then
			time_tmp = time.time() * 1000
			print('Time elapse Provisioning-Staging: ', provisioning)
			flag = 0

		if result['status'] == 'RUNNING':
			now = time.time() * 1000
			nownow = str(datetime.utcnow())
			# print "Running timestamp: ", now
			if 'error' in result:
			    raise Exception(result['error'])
			staging = now - time_tmp
			print('Time stamp of Running: ', nownow)
			print('Time elapse Staging-Running: ', staging)
			return True

def check_delete_process(zone, name):
	compute = googleapiclient.discovery.build('compute', 'v1')
	project = "shijian-18"
	flag = 1
	while True:
		try:
			request = compute.instances().get(project=project, zone=zone, instance=name)
			result = request.execute()
			# print result
			if result['status'] == 'STOPPING' and flag == 1:
				then = time.time() * 1000
				if 'error' in result:
					raise Exception(result['error'])
				flag = 0

			if result['status'] == 'TERMINATED' and flag == 0:
				now = time.time() * 1000
				nownow = str(datetime.utcnow())
				if 'error' in result:
					raise Exception(result['error'])
				stopping = now - then
				time_tmp = time.time() * 1000
				print('Stop timestamp: ', nownow)
				print('Time elapse stopping an instance: ', stopping)
				flag = -1


		except googleapiclient.errors.HttpError:
			now2 = time.time() * 1000
			nownow2 = str(datetime.utcnow())
			deleting = now2 - time_tmp
			print('Deletion timestamp: ', nownow2)
			print('Time elapse deleting an instance: ', deleting)
			return True

# Run the server's main loop
server.serve_forever()