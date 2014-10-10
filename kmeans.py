import numpy as np
# import sklearn
# import csv
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans, Ward
from sklearn import metrics

# def choosing_k():
# 	train = np.loadtxt("1.csv",delimiter = ',')
# 	test = np.loadtxt("1_test.csv",delimiter = ',')
		
# 	X = train[:,0:21]
# 	Y = train[:,22]
	
# 	k_range = range(1,95)

# 	k_means_var = [KMeans(n_clusters = k).fit(X,Y) for k in k_range]

# 	centroids = [C.cluster_centers_ for C in k_means_var]

# 	k_euclid = [cdist(X, cent, 'euclidean') for cent in centroids]

# 	dist = [np.min(ke, axis = 1) for ke in k_euclid]

# 	wcss = [sum(d**2) for d in dist]

# 	tss = sum(pdist(X) ** 2)/X.shape[0]

# 	bss = wcss - tss

# 	print bss.argmin()


def matr_analys(pred_mat):
	perc = np.empty([pred_mat.shape[0],1], dtype = float)

	for i in range(0,pred_mat.shape[0]):
		err = 0.0
		for j in range(0,pred_mat.shape[1]):
			if pred_mat[i,j] == 0:
				err += 1.0
		perc[i,0] = round(float((err/pred_mat.shape[1])),3)
	# print perc
	pred_mat = np.hstack((pred_mat,perc))
	
	error =0.0
	for i in range(0,pred_mat.shape[0]):
		if(pred_mat[i,pred_mat.shape[1]-1] > .50):
			error += 1.0
	
	print float(error/pred_mat.shape[0])
	
	# print pred_mat
	return pred_mat

def main():
	pred_mat = np.empty([95, 1], dtype = bool)
	
	for i in range(1,2):
		train = np.loadtxt(str(i)+".csv",delimiter = ',')
		test = np.loadtxt(str(i)+"_test.csv",delimiter = ',')
		
		X = train[:,0:21]
		Y = train[:,22]
		 

		t1 = test[:,0:21]
		t1_labels = test[:,22]

		clf = Ward(n_clusters = 95)
		t = clf.fit(X)

		# pred = t.predict(X)
		print t.labels_#metrics.accuracy_score(pred, Y)
		# clf = KMeans(n_clusters = 95)
		# t = clf.fit(X,Y)
		
		# S = pairwise_distances(t1, clf.cluster_centers_)
		# pred = t.predict(X)
		# print metrics.accuracy_score(pred, Y)
		# pred = np.equal(t1_labels, S.argmin(axis = 1))
		# print pred
		# h = pred.reshape(95,1)
		
		# pred_mat = np.hstack((pred_mat,h))
		i += 1
	
	# pred_mat = matr_analys(pred_mat)
	# print pred_mat
	# np.savetxt("kmeans_test_res.txt",pred_mat[:,1:],fmt = "%.2f", delimiter= ',')

if __name__ == '__main__':
	main()
	# for i in range(8,12):
	# 	main(i)
	# choosing_k()