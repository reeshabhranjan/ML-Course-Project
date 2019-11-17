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
			if not np.isnan(new_row).any():
				normal_timeseries[i - merging_extent] = new_row
		normal_timeseries_list.append(normal_timeseries)

	for aggressive in aggressive_list:
		aggressive_timeseries = np.zeros((aggressive.shape[0] - merging_extent, aggressive.shape[1] * (merging_extent + 1)))
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

	for i, normal_timeseries in enumerate(normal_timeseries_list):
		normal_timeseries_list[i] = np.concatenate((normal_timeseries, np.full((normal_timeseries.shape[0], 1), 0)), axis=1)
		
	for i, aggressive_timeseries in enumerate(aggressive_timeseries_list):
		aggressive_timeseries_list[i] = np.concatenate((aggressive_timeseries, np.full((aggressive_timeseries.shape[0], 1), 1)), axis=1)
	
	for i, drowsy_timeseries in enumerate(drowsy_timeseries_list):
		drowsy_timeseries_list[i] = np.concatenate((drowsy_timeseries, np.full((drowsy_timeseries.shape[0], 1), 2)), axis=1)