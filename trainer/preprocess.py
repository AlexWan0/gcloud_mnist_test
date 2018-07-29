from __future__ import print_function

import _pickle as pickle
import numpy as np
import gzip
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train-files',
	required=True,
	type=str,
	help='Data zip path')
args = parser.parse_args()
data_path = args.train_files

'''
get_data type:
0 = train
1 = validation
2 = test
'''
def get_data(type_num):
	# get file from gcloud bucket
	compressed_data = cloudstorage.open(data_path, 'rb')

	# unzip data file
	unzipped_file = gzip.GzipFile(fileobj=compressed_data)

	#following is for running locally
	#file = gzip.open(data_path, "rb")
	
	# gets data from pickle file
	# returns tuple (train, validation, test)
	data = pickle.load(unzipped_file, encoding='latin1')

	x, y = data[type_num]

	y = to_one_hot(10, y)

	file.close()

	return x, y

def test_data():
	# opens pickled mnist dataset and splits resulting data
	file = gzip.open(data_path, "rb")
	
	train, validation, test = pickle.load(file, encoding='latin1')

	test_x, test_y = test

	test_y = to_one_hot(10, test_y)

	file.close()

	return test_x, test_y

def to_one_hot(num_classes, data):
	a = np.array(data)
	b = np.zeros((len(data), num_classes))
	b[np.arange(len(data)), a] = 1
	return b