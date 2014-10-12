import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# np.set_printoptions(threshold=np.nan)
		
def sel_col(t, t_argsort, col_num, num_gest,k):
	t_min2 = t_argsort[:,95-col_num:95]
	return find_closest_match(t, t_min2, t_argsort.shape[0], num_gest, k)

def find_closest_match(t, t_min, row, num_gest, k):
	count = 0
	per_frame = np.empty((t.shape[1],num_gest,num_gest))
	# Define purpose of per_frame
	# print per_frame.shape
	
	# for misprediction rate at the frames:
	for i in range(0,row):
		for j in range(t_min.shape[1]-1,-1,-1):
			if (i != t_min[i,j]): # for the closest neighbour match; earlier work
				if (t_min.shape[1]-1-j > num_gest):
					# t_row = t[i:i+1,j:t_min.shape[1]]
					# print i, t_min[i,j:t_min.shape[1]], t_row.min(), k
					count += 1
					break
			else:
				break
	# print "For the ",k,"th frame, the incorrect predictions is: ", count,"with the misprediction rate at:" ,float(count)/t_min.shape[0]
	

	for i in range(0,row):	
		for l in range(t_min.shape[1]-1,t_min.shape[1]-1-num_gest,-1):
			per_frame[i,t_min.shape[1]-1-l,0] = t_min[i,l]
			per_frame[i,t_min.shape[1]-1-l,1] = t[i,t_min[i,l]]

	# per_frame is the matrix which stores for each frame, the nearest gestrue and the probability assigned to that gesture
	return per_frame, float(count)/t_min.shape[0]
	
def PCA_trans(X, n_co):
	pca = PCA(n_components = n_co)
	return pca.fit(X)

def initialize(num_gest, col_num, n_co_PCA):
	# num_gest = 2
	# col_num = 10
	init_frame = np.zeros((95,num_gest,num_gest))
	GNB = []
	INCP = [] # for incorrect prediction graphing
	K = [] # for value of k
	pca_per_frame = []
	for k in range (1,21):
		train = np.loadtxt(str(k)+".csv",delimiter = ",")

		X = train[:,0:21]
		Y = train[:,22]

		# nbclf = GaussianNB()
		# nbclf.fit(X,Y)
		
		# Uncomment the above lines and comment the below lines to remove PCA
		pca_X = PCA_trans(X, n_co_PCA)
		pca_per_frame.append(pca_X)

		X_trans = pca_X.transform(X)

		nbclf = GaussianNB()
		nbclf.fit(X_trans,Y)
		
		y_pred_proba = nbclf.predict_proba(X_trans)
		t = np.zeros((95,95), dtype = float)
		# Define t here.

		for i in range(0,X_trans.shape[0]):
			cl = int(Y[i,]) # Define purpose of cl here
			t[cl:cl+1,:] += nbclf.class_prior_[cl] * y_pred_proba[i:i+1,:]
		
		# At the end of this loop here, what is t holding right now?
		
		frame_pred, in_pr = sel_col(t, t.argsort(axis = 1), col_num, num_gest, k)
		init_frame = np.hstack((init_frame,frame_pred))
		GNB.append(nbclf)
		
		INCP.append(in_pr)
		K.append(k)

	init_frame = init_frame[:,num_gest:,:]
	
	return init_frame, GNB, num_gest, INCP, K, pca_per_frame

def classify(trained_model, clf, num_gest, pca_per_frame):
	class_weight = np.zeros((95,95), dtype = float)
	for k in range(1,21):

		test = np.loadtxt(str(k)+"_test.csv",delimiter = ',')
		t1 = test[:,0:21] 

		t1_labels = test[:,22]

		t1_trans = pca_per_frame[k-1].transform(t1)
		# storing the probability predictions.
		proba_pred = clf[k-1].predict_proba(t1_trans)

		for j in range(0,proba_pred.shape[0]):
			max_index = proba_pred[j].argmax()
			max_proba = proba_pred[j].max()

			nearest_gest = trained_model[max_index]
			for i in range(num_gest):
				class_weight[nearest_gest[2*(k-1)+i,0].astype(int),j] += (max_proba * nearest_gest[2*(k-1)+i,1])

	pred_labels = np.argmax(class_weight, axis = 0)
	
	return metrics.accuracy_score(t1_labels.astype(int), pred_labels)

# Return the top 2 most predicted rows...
def return_top_two(proba_pred_row):
	max_index_1 = proba_pred_row.argmax()
	max_proba_1 = proba_pred_row.max()
	proba_pred_row[max_index_1] = 0.0
	max_index_2 = proba_pred_row.argmax()
	max_proba_2 = proba_pred_row.max()

	# print max_index_1, max_proba_1, max_index_2, max_proba_2
	return max_index_1, max_proba_1, max_index_2, max_proba_2


def classify_2(trained_model, clf, num_gest):
	print "Entering testing phase: "
	class_weight = np.zeros((95,95), dtype = float)
	# The second arguement of class_weight i.e. the Cols are supposed to be equal to the number of testing examples.
	
	class_weight_pf = np.zeros((95,1), dtype = float)
	for k in range(1,21):
		test = np.loadtxt(str(k)+"_test.csv",delimiter = ',')
		t1 = test[:,0:21] 

		t1_labels = test[:,22]

		proba_pred = clf[k-1].predict_proba(t1)
		# print proba_pred
		for j in range(0,proba_pred.shape[0]):
			class_weight_pf.fill(0.0)
			mi_1, mp_1, mi_2, mp_2 = return_top_two(proba_pred[0])

			class_weight_pf[trained_model[mi_1,2*(k-1)+0,0].astype(int), 0] += (mp_1 * trained_model[mi_1,2*(k-1)+0,1])
			class_weight_pf[trained_model[mi_1,2*(k-1)+1,0].astype(int), 0] += (mp_1 * trained_model[mi_1,2*(k-1)+1,1])
			class_weight_pf[trained_model[mi_2,2*(k-1)+0,0].astype(int), 0] += (mp_2 * trained_model[mi_2,2*(k-1)+0,1])
			class_weight_pf[trained_model[mi_2,2*(k-1)+1,0].astype(int), 0] += (mp_2 * trained_model[mi_2,2*(k-1)+1,1])

			cw_1,cwp_1,cw_2,cwp_2 = return_top_two(class_weight_pf)

			class_weight[cw_1,j] += cwp_1
			class_weight[cw_2,j] += cwp_2

		pred_labels = np.argmax(class_weight, axis = 0)
	
		print metrics.accuracy_score(t1_labels.astype(int), pred_labels), k


if __name__ == '__main__':
	num_gest = 2
	col_num = 21
	for i in range(21,22):
		n_co_PCA = i #10
		print 'Number of components saved are :', i
		trained_model, clf, num_gest, INCP, K, pca_per_frame = initialize(num_gest, col_num, n_co_PCA)
		print classify(trained_model, clf, num_gest, pca_per_frame)

# 	Below to be used for plotting.
# 	
# if __name__ == '__main__':
# 	# error = []
# 	# gest = []
# 	# M_INCP = []
# 	# c = ['r','b','g','c','m','y','k','w']
# 	# for i in range(2,10):
# 	# 	print i,'th round'
	
# 	num_gest = 3
# 	# gest.append(num_gest)
# 	col_num = 10
# 	trained_model, clf, num_gest, INCP, K = initialize(num_gest, col_num)
# 	print classify(trained_model, clf, num_gest)
# 	# classify_2(trained_model, clf, num_gest)

# 	# 	plt.plot(K, INCP, marker = 'o', color = c[i-2], linestyle = '--', label = str(i)+" Gestures")
# 	# plt.legend()
# 	# plt.show()
# 	# error.append(classify(trained_model, clf, num_gest))
	
	
# 	# plt.plot(gest, error)
# 	# plt.xlabel("Number of Gestures used for scoring.")
# 	# plt.ylabel("Accuracy")
# 	# plt.show()