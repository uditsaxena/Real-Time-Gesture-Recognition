# This is different than the file: nbclass.py as 
# the way the model is constructed at each time frame is
# different.

import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt

# np.set_printoptions(threshold=np.nan)
		
def initialize():
	
	init_frame = np.zeros((95, 95, 95)) # This 95 is the number of classes known prior
	
	GNB = []
	Model = []
	# Storing the Gaussian NBCs ^
	
	for k in range (1,21):
		train = np.loadtxt(str(k)+".csv",delimiter = ",")

		X = train[:,0:21]
		Y = train[:,22]

		nbclf = GaussianNB()
		nbclf.fit(X,Y)
		
		y_pred_proba = nbclf.predict_proba(X)
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
	
	return Model, GNB

def classify(trained_model, clf, top_gest):
	class_weight = np.zeros((95,95), dtype = float)
	for k in range(1,21):
		n_gest = trained_model[k-1]
		# n_gest is the matrix of the corresponding time-frame
		# which holds the class predictions for the time-frame.
		
		test = np.loadtxt(str(k)+"_test.csv",delimiter = ',')
		t1 = test[:,0:21] 

		t1_labels = test[:,22]

		# storing the probability predictions.
		proba_pred = clf[k-1].predict_proba(t1)
		max_indexes = proba_pred.argsort(axis = 1)

		for j in range(0,proba_pred.shape[0]):
			top = max_indexes[95-top_gest:95]	
			# top stores the top 'x' gestures using the top_gest var.

			for i in range(top.shape[1]):
				class_weight[top[i],j] += (n_gest[] * nearest_gest[2*(k-1)+i,1])

	pred_labels = np.argmax(class_weight, axis = 0)
	
	return metrics.accuracy_score(t1_labels.astype(int), pred_labels)


if __name__ == '__main__':
	
	top_gest = 3 # the top 'x' gestures to consider.
	trained_model, clf = initialize()
	classify(trained_model, clf, top_gest)
	