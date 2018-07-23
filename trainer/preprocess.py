from __future__ import print_function

import pickle as pickle
import numpy as np
import gzip
def train_data():
	# opens pickled mnist dataset and splits resulting data
	file = gzip.open("../data/mnist.pkl.gz", "rb")
	
	train, validation, test = pickle.load(file, encoding='latin1')

	train_x, train_y = train

	train_y = to_one_hot(10, train_y)

	file.close()

	return train_x, train_y

def test_data():
	# opens pickled mnist dataset and splits resulting data
	file = gzip.open("../data/mnist.pkl.gz", "rb")
	
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