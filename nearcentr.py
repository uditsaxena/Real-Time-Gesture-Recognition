import numpy as np
# import sklearn
# import csv
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.metrics.pairwise import pairwise_distances


def main():
	pred_mat = np.empty([95, 1], dtype = bool)
	
	for i in range(1,41):
		train = np.loadtxt(str(i)+".csv",delimiter = ',')
		test = np.loadtxt(str(i)+"_test.csv",delimiter = ',')
		
		X = train[:,0:21]
		Y = train[:,22]
		
		t1 = test[:,0:21]
		t1_labels = test[:,22]

		clf = NearestCentroid()
		clf.fit(X,Y)
		
		# print clf.centroids_,'\n\n'
		S = pairwise_distances(t1, clf.centroids_)
		
		pred = np.equal(t1_labels, S.argmin(axis = 1))
		
		h = pred.reshape(95,1)
		
		pred_mat = np.hstack((pred_mat, h))
		
		i += 1

	print pred_mat

if __name__ == '__main__':
	main()