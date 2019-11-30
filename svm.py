import data_extractor
import numpy as np
import sklearn
import sklearn.preprocessing
import sklearn.model_selection
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sys import exit

if __name__ == '__main__':

	x_train, x_test, y_train, y_test = data_extractor.prepare_dataset_for_svm()

	# clf = RandomForestClassifier(n_estimators=50, random_state=2)
	clf = SVC(gamma='auto', kernel='rbf', verbose=True, probability=True)
	clf.fit(x_train, y_train)
	print("Train accuracy: " + str(clf.score(x_train, y_train)))
	y_pred = clf.predict(x_test)
	print("Test accuracy: " + str(clf.score(x_test, y_test)))