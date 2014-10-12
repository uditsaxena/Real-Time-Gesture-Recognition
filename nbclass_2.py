# This is different than the file: nbclass.py as 
# the way the model is constructed at each time frame is
# different.

import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
np.set_printoptions(threshold=np.nan)

def PCA_trans(X, n_co):
	pca = PCA(n_components = n_co)
	return pca.fit(X)
		
def initialize(num_files, n_co_PCA):

	GNB = []
	Model = []
	pca_per_frame = []
	
	for k in range (1,num_files):
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
		# The rows and cols of t are the number of classes.
		# Each class is mapped to every other class.

		for i in range(0,X.shape[0]):
			cl = int(Y[i,]) # cl is the true class of the i-th row.
			t[cl:cl+1,:] += nbclf.class_prior_[cl] * y_pred_proba[i:i+1,:]
		
		# At the end of this loop, t holds, for this frame:
		# 	each row is a class which maps cumulative weights to the nearest 
		# 	predicted class which are the columns.
		
		# Instead of building a 3-D matrix, just keeping a track of the t 
		# matrices built at each time-frame, just like the GNB
		Model.append(t)
		GNB.append(nbclf)
	
	return Model, GNB, pca_per_frame

# utility fucntion to get labesl for previous classes.
# def get_prev_class():
	

def classify(num_files, trained_model, clf, top_gest, pca_per_frame):
	class_weight = np.zeros((95,95), dtype = float)
	# Here the number of the columns are the number of test records 
	
	coming_from = []
	# coming from holds the class labels held in the previous frame
	# each matrix shape is of the form: number of test records x num of top_gest
	# initially num of frames is 0.
	
	for k in range(1,num_files):
		n_gest = trained_model[k-1]
		# n_gest is the matrix of the corresponding time-frame
		# which holds the class predictions for the time-frame.
		
		test = np.loadtxt(str(k)+"_test.csv",delimiter = ',')
		t1 = test[:,0:21] 

		t1_labels = test[:,22]
		t1_trans = pca_per_frame[k-1].transform(t1)
		# storing the probability predictions.
		proba_pred = clf[k-1].predict_proba(t1_trans)
		max_indexes = proba_pred.argsort(axis = 1)
		proba_pred_sorted = np.sort(proba_pred,axis = 1)
		
		top = max_indexes[:,95-top_gest:95]	
		top_proba = proba_pred_sorted[:,95-top_gest:95]
		
		if (k == 1):
			# This means we are at the beginning of the frames
			# No previous classes yet, just go normally to the top_gest classes
			for j in range(0,proba_pred.shape[0]):
				for i in range(top.shape[1]):
					class_weight[top[j,i],j] += top_proba[j,i] #proba_pred[j,top[j,i]]
					# Here we are not taking the input from where the class is coming,
					# because this is the starting time frame. That happens later.
							
		else:
			# Not at the beginning of frames, now start to use the prev classes from 
			# which you've arrived from.
			prev_classes = coming_from[k-2]
			for j in range(0,proba_pred.shape[0]):
				for i in range(top.shape[1]):
					class_to = top[j,i]
					
					for ki in range(prev_classes.shape[1]):
						class_from = prev_classes[j,ki]
						
						class_weight[class_to,j] += (n_gest[class_from,class_to]) * proba_pred[j,class_to]
				
			# top stores the top 'x' gestures using the top_gest var.
		# print class_weight
		coming_from.append(top)

	pred_labels = np.argmax(class_weight, axis = 0)
	# print class_weight, pred_labels
	return metrics.accuracy_score(t1_labels.astype(int), pred_labels)


if __name__ == '__main__':
	top_gest = 2 # the top 'x' gestures to consider.
	frames_to_consider = 21
	n_co_PCA = 18 # got this number through analysis.
	trained_model, clf, pca_per_frame = initialize(frames_to_consider, n_co_PCA) 
	print classify(frames_to_consider, trained_model, clf, top_gest, pca_per_frame)

# Use the below for plotting.
# 
# if __name__ == '__main__':
# 	c = ['r','b','g','c','m','y','k','w','#b3de69','#fa8174','#00FFCC','#6d904f']
# 	for i in range(3,15):
# 		print 'i',i 
# 		frames = []
# 		classify_rate = []
# 		for j in range(10,41):
# 			print 'j', j
# 			top_gest = i # the top 'x' gestures to consider.
# 			trained_model, clf = initialize(j)
# 			# print i,'-',j,':',classify(j, trained_model, clf, top_gest)
# 			frames.append(j)
# 			classify_rate.append(classify(j, trained_model, clf, top_gest))

# 		plt.plot(frames, classify_rate, marker = 'o', color = c[i-3], linestyle = '--', label = str(i)+"NN Gestures")
# 	plt.legend()
# 	plt.show()