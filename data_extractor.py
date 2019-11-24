import numpy as np
import os
import glob
import sklearn
import sklearn.preprocessing
import sklearn.utils

data_path_root = 'UAH-DRIVESET-v1'


def load_all_data():
	"""Loads data and returns list of datasets as list.
	List is segmented as behaviour type.
	List elements: [normal, aggressive, drowsy]."""

	normal_list = [] # contains data in form of numpy matrices corresponding to normal behaviour
	aggressive_list = [] # for aggressive behaviour
	drowsy_list = [] # similarly for drowsy behaviour

	file_list = glob.glob('**/SEMANTIC_ONLINE.txt', recursive=True)  # all SEMANTIC_ONLINE.txt files using regex

	# fix the slashes
	for i, filename in enumerate(file_list):
		file_list[i] = filename.replace("\\", "/")

	# load the data
	for filename in file_list:
		if "NORMAL" in filename:
			normal_list.append(np.loadtxt(filename))
		elif "DROWSY" in filename:
			drowsy_list.append(np.loadtxt(filename))
		elif "AGGRESSIVE" in filename:
			aggressive_list.append(np.loadtxt(filename))

	# for normal in _normal:
	# 	print(normal.shape[0])

	# filtering nan values, reference: https://stackoverflow.com/a/11453235/5394180
	for i, normal in enumerate(normal_list):
		normal_list[i] = normal[~np.isnan(normal).any(axis=1)]
	for i, aggressive in enumerate(aggressive_list):
		aggressive_list[i] = aggressive[~np.isnan(aggressive).any(axis=1)]
	for i, drowsy in enumerate(drowsy_list):
		drowsy_list[i] = drowsy[~np.isnan(drowsy).any(axis=1)]

	return normal_list, aggressive_list, drowsy_list

def prepare_dataset_for_lstm():
	normal_list, aggressive_list, drowsy_list = load_all_data()

	timestamp_dimension = 60

	global_x = np.empty((0, timestamp_dimension, normal_list[0].shape[1] - 1))
	global_y = np.empty((0, 1))

	for i, normal in enumerate(normal_list):
		normal = normal[:, 1:]
		normal = sklearn.preprocessing.normalize(normal)
		for j in range(normal.shape[0] // timestamp_dimension):
			sample = normal[j * timestamp_dimension : j * timestamp_dimension + timestamp_dimension]
			# remove the time column
			# sample = sample[:, 1:]
			# normalise this batch
			# sample = sklearn.preprocessing.normalize(sample)
			sample = np.asarray([sample])
			# sample = np.moveaxis(np.atleast_3d(sample), -1, 0)
			global_x = np.append(global_x, sample, axis=0)
			global_y = np.append(global_y, np.full((1, 1), 0), axis=0)

	for i, aggressive in enumerate(aggressive_list):
		aggressive = aggressive[:, 1:]
		aggressive = sklearn.preprocessing.normalize(aggressive)
		for j in range(aggressive.shape[0] // timestamp_dimension):
			sample = aggressive[j * timestamp_dimension : j * timestamp_dimension + timestamp_dimension]
			sample = np.asarray([sample])
			# sample = np.moveaxis(np.atleast_3d(sample), -1, 0)
			global_x = np.append(global_x, sample, axis=0)
			global_y = np.append(global_y, np.full((1, 1), 1), axis=0)
			
	for i, drowsy in enumerate(drowsy_list):
		drowsy = drowsy[:, 1:]
		drowsy = sklearn.preprocessing.normalize(drowsy)
		for j in range(drowsy.shape[0] // timestamp_dimension):
			sample = drowsy[j * timestamp_dimension : j * timestamp_dimension + timestamp_dimension]
			sample = np.asarray([sample])
			# sample = np.moveaxis(np.atleast_3d(sample), -1, 0)
			global_x = np.append(global_x, sample, axis=0)
			global_y = np.append(global_y, np.full((1, 1), 2), axis=0)

	return global_x, global_y

if __name__ == '__main__':
	# normal, aggressive, drowsy = load_all_data()
	x, y = prepare_dataset_for_lstm()
	x, y = sklearn.utils.shuffle(x, y, random_state=2)
	print(x.shape)
	print(y.shape)
	np.save("x", x)
	np.save("y", y)
