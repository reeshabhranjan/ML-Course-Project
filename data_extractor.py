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
			sample = np.asarray([sample])
			global_x = np.append(global_x, sample, axis=0)
			global_y = np.append(global_y, np.full((1, 1), 0), axis=0)

	for i, aggressive in enumerate(aggressive_list):
		aggressive = aggressive[:, 1:]
		aggressive = sklearn.preprocessing.normalize(aggressive)
		for j in range(aggressive.shape[0] // timestamp_dimension):
			sample = aggressive[j * timestamp_dimension : j * timestamp_dimension + timestamp_dimension]
			sample = np.asarray([sample])
			global_x = np.append(global_x, sample, axis=0)
			global_y = np.append(global_y, np.full((1, 1), 1), axis=0)
			
	for i, drowsy in enumerate(drowsy_list):
		drowsy = drowsy[:, 1:]
		drowsy = sklearn.preprocessing.normalize(drowsy)
		for j in range(drowsy.shape[0] // timestamp_dimension):
			sample = drowsy[j * timestamp_dimension : j * timestamp_dimension + timestamp_dimension]
			sample = np.asarray([sample])
			global_x = np.append(global_x, sample, axis=0)
			global_y = np.append(global_y, np.full((1, 1), 2), axis=0)

	return global_x, global_y

def prepare_dataset_for_svm():
	normal_list, aggressive_list, drowsy_list = load_all_data()
	normal_timeseries_list = []  # this will contain list of datasets in form of numpy matrices
	# that will include time-based data (by appending last 'merging-extent'
	# rows to a row
	aggressive_timeseries_list = []  # similarly for aggressive
	drowsy_timeseries_list = []  # similarly for drowsy

	merging_extent = 10  # This tells for ith row, how many previous rows are we appending to it to
	# make our ML model aware of the time-based interactions

	# the following six for-loops just prepares the lists defined above
	for normal in normal_list:
		normal_timeseries = np.zeros((normal.shape[0] - merging_extent, normal.shape[1] * (merging_extent + 1)))
		for i in range(merging_extent, normal.shape[0]):
			new_row = normal[i]
			for j in range(1, merging_extent + 1):
				new_row = np.append(new_row, normal[i - j])
			if not np.isnan(new_row).any():
				normal_timeseries[i - merging_extent] = new_row
		normal_timeseries_list.append(normal_timeseries)

	for aggressive in aggressive_list:
		aggressive_timeseries = np.zeros(
			(aggressive.shape[0] - merging_extent, aggressive.shape[1] * (merging_extent + 1)))
		for i in range(merging_extent, aggressive.shape[0]):
			new_row = aggressive[i]
			for j in range(1, merging_extent + 1):
				new_row = np.append(new_row, aggressive[i - j])
			if not np.isnan(new_row).any():
				aggressive_timeseries[i - merging_extent] = new_row
		aggressive_timeseries_list.append(aggressive_timeseries)

	for drowsy in drowsy_list:
		drowsy_timeseries = np.zeros((drowsy.shape[0] - merging_extent, drowsy.shape[1] * (merging_extent + 1)))
		for i in range(merging_extent, drowsy.shape[0]):
			new_row = drowsy[i]
			for j in range(1, merging_extent + 1):
				new_row = np.append(new_row, drowsy[i - j])
			if not np.isnan(new_row).any():
				drowsy_timeseries[i - merging_extent] = new_row
		drowsy_timeseries_list.append(drowsy_timeseries)

	# assign labels
	for i, normal_timeseries in enumerate(normal_timeseries_list):
		normal_timeseries_list[i] = np.concatenate((normal_timeseries, np.full((normal_timeseries.shape[0], 1), 0)),
		                                           axis=1)

	for i, aggressive_timeseries in enumerate(aggressive_timeseries_list):
		aggressive_timeseries_list[i] = np.concatenate(
			(aggressive_timeseries, np.full((aggressive_timeseries.shape[0], 1), 1)), axis=1)

	for i, drowsy_timeseries in enumerate(drowsy_timeseries_list):
		drowsy_timeseries_list[i] = np.concatenate((drowsy_timeseries, np.full((drowsy_timeseries.shape[0], 1), 2)),
		                                           axis=1)

	# finally I will concatenate all the data into one dataset
	dataset = normal_timeseries_list[0]

	for i in range(1, len(normal_timeseries_list)):
		dataset = np.concatenate((dataset, normal_timeseries_list[i]), axis=0)

	for i in range(len(aggressive_timeseries_list)):
		dataset = np.concatenate((dataset, aggressive_timeseries_list[i]), axis=0)

	for i in range(len(drowsy_timeseries_list)):
		dataset = np.concatenate((dataset, drowsy_timeseries_list[i]), axis=0)

	# shuffle the dataset
	np.random.shuffle(dataset)

	# we do not need the time column anymore
	X = dataset[:, 1:-1]
	Y = dataset[:, -1]

	# normalise the dataset
	X = sklearn.preprocessing.normalize(X)

	x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=2)
	return x_train, x_test, y_train, y_test

if __name__ == '__main__':
	pass
