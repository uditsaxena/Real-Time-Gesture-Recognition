import numpy as np
from sklearn import metrics
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.multiclass import OneVsRestClassifier

def main():
	
	# for i in range(1,41):
	train = np.loadtxt("1.csv",delimiter = ',')
	test = np.loadtxt("1_test.csv",delimiter = ',')
	
	X = train[:,0:21]
	Y = train[:,22]
	
	t1 = test[:,0:21]
	t1_labels = test[:,22]

	clf = NearestCentroid()
	clf.fit(X,Y)

	y_pred = clf.predict(X)

	print metrics.accuracy_score(y_pred,Y)

	test_pred = clf.predict(t1)

	print metrics.accuracy_score(test_pred, t1_labels)

	
		# print clf.centroids_,'\n\n'
		# S = pairwise_distances(t1, clf.centroids_)
		
		# pred = np.equal(t1_labels, S.argmin(axis = 1))
		
		# h = pred.reshape(95,1)
		
		# pred_mat = np.hstack((pred_mat, h))
		
		# i += 1

	# pred_mat = matr_analys(pred_mat)
	# print pred_mat
	# np.savetxt("nearcentr_test_res.txt",pred_mat[:,1:],fmt = "%.2f", delimiter= ',')

if __name__ == '__main__':
	main()