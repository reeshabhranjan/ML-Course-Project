import numpy as np
import os
import glob

data_path_root = 'UAH-DRIVESET-v1'


def load_all_data():
	"""Loads data and returns list of datasets as list.
	List is segmented as behaviour type.
	Code referred from: https://stackoverflow.com/a/19587581"""

	normal = []
	aggressive = []
	drowsy = []

	file_list = glob.glob('**/SEMANTIC_ONLINE.txt', recursive=True)  # all SEMANTIC_ONLINE.txt files using regex

	# fix the slashes
	for i, filename in enumerate(file_list):
		file_list[i] = filename.replace("\\", "/")

	# load the data
	for filename in file_list:
		if "NORMAL" in filename:
			normal.append(np.loadtxt(filename))
		elif "DROWSY" in filename:
			drowsy.append(np.loadtxt(filename))
		elif "AGGRESSIVE" in filename:
			aggressive.append(np.loadtxt(filename))

	return normal, aggressive, drowsy


if __name__ == '__main__':
	normal, aggressive, drowsy = load_all_data()
