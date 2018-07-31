from __future__ import print_function

import _pickle as pickle
import numpy as np
import gzip
import argparse
from google.cloud import storage

# gets command line arguments
parser = argparse.ArgumentParser()

# gets url to bucket
parser.add_argument('--bucket-url',
	required=True,
	type=str,
	help='Url to bucket')

# gets path to dataset on bucket
parser.add_argument('--dataset-path',
	required=True,
	type=str,
	help='Path to dataset on bucket')

args = parser.parse_args()

bucket_url = args.bucket_url
dataset_path = args.dataset_path

'''
get_data type:
0 = train
1 = validation
2 = test
'''
def get_data(type_num):
	# cloud storage client
	gcs = storage.Client()

	# get bucket
	bucket = gcs.get_bucket(bucket_url)

	# get file from bucket as blob
	dataset_file = bucket.blob(dataset_path)

	# unzips file
	unzipped_file = gzip.GzipFile(fileobj=dataset_file, mode="rb")

	# following is for using local dataset
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