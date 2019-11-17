import data_extractor
import numpy as np

if __name__ == '__main__':
	normal_list, aggressive_list, drowsy_list = data_extractor.load_all_data()
	normal_timeseries_list = []
	aggressive_timeseries_list = []
	drowsy_timeseries_list = []

	merging_extent = 5

	for normal in normal_list:
		normal_timeseries = np.zeros((normal.shape[0] - merging_extent, normal.shape[1] * (merging_extent + 1)))
		for i in range(merging_extent, normal.shape[0]):
			new_row = normal[i]
			for j in range(1, merging_extent + 1):
				new_row = np.append(new_row, normal[i - j])
			normal_timeseries[i - merging_extent] = new_row
		normal_timeseries_list.append(normal_timeseries)


	# normal_timeseries_list = np.zeros((normal_list[0].shape[0] - 5, 27 * 6))
	# for i in range(5, normal_list[0].shape[0]):
	# 	new_row = normal_list[0][i]
	# 	new_row = np.append(new_row, normal_list[0][i - 1])
	# 	new_row = np.append(new_row, normal_list[0][i - 2])
	# 	new_row = np.append(new_row, normal_list[0][i - 3])
	# 	new_row = np.append(new_row, normal_list[0][i - 4])
	# 	new_row = np.append(new_row, normal_list[0][i - 5])
	# 	print(i)
	# 	normal_timeseries_list[i - 5] = new_row
