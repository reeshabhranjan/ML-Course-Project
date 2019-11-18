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

	return _normal, _aggressive, _drowsy


if __name__ == '__main__':
	normal, aggressive, drowsy = load_all_data()
