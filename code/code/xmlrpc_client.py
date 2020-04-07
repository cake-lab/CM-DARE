import xmlrpclib
import argparse

def main():
	s = xmlrpclib.ServerProxy('http://'+FLAGS.ip+':8000')
	if FLAGS.input == 'setMaster':
		print s.setMaster(FLAGS.type, FLAGS.index)
		print "Master node has changed to:" + str(FLAGS.type)+ ", " + str(FLAGS.index)
	elif FLAGS.input == 'showMaster':
		print s.getMaster()
	elif FLAGS.input == 'showWorkers':
		workers = s.getWorkers()
		for i in workers:
			print(i)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
			'--ip',
	        type=str,
	        required=True,
	        help='IP of RPC server')
	parser.add_argument(
	        '--input',
	        type=str,
	        required=True,
	        help='showWorkers, getMaster, setMaster')
	parser.add_argument(
	        '--index',
	        type=str,
	        required=True,
	        help='index number in job for cluster')
	parser.add_argument(
	        '--type',
	        type=str,
	        required=True,
	        help='index number in job for cluster')
	FLAGS, unparsed = parser.parse_known_args()
	main()
