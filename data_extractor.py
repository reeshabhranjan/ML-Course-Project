import numpy as np
import os
import glob

data_path_root = 'UAH-DRIVESET-v1'


def load_all_data():
	"""Loads data and returns list of datasets as list.
	List is segmented as behaviour type.
	List elements: [normal, aggressive, drowsy]."""

	_normal = [] # contains data in form of numpy matrices corresponding to normal behaviour
	_aggressive = [] # for aggressive behaviour
	_drowsy = [] # similarly for drowsy behaviour

	file_list = glob.glob('**/SEMANTIC_ONLINE.txt', recursive=True)  # all SEMANTIC_ONLINE.txt files using regex

	# fix the slashes
	for i, filename in enumerate(file_list):
		file_list[i] = filename.replace("\\", "/")

	# load the data
	for filename in file_list:
		if "NORMAL" in filename:
			_normal.append(np.loadtxt(filename))
		elif "DROWSY" in filename:
			_drowsy.append(np.loadtxt(filename))
		elif "AGGRESSIVE" in filename:
			_aggressive.append(np.loadtxt(filename))

	# for normal in _normal:
	# 	print(normal.shape[0])

	return _normal, _aggressive, _drowsy

def prepare_dataset_for_lstm():
	normal_list, aggressive_list, drowsy_list = load_all_data()

	timestamp_dimension = 100

	global_array = np.zeros((0, timestamp_dimension, normal_list[0].shape[1]))

	for i, normal in enumerate(normal_list):
		for j in range(normal.shape[0] // timestamp_dimension):
			sample = normal[j * timestamp_dimension : j * timestamp_dimension + timestamp_dimension]
			sample = np.asarray([sample])
			# sample = np.moveaxis(np.atleast_3d(sample), -1, 0)
			global_array = np.append(global_array, sample, axis=0)

	print(global_array.shape)

if __name__ == '__main__':
	# normal, aggressive, drowsy = load_all_data()
	prepare_dataset_for_lstm()