import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from sklearn.multiclass import OneVsRestClassifier

def initialhc():
	train = np.loadtxt("1.csv",delimiter = ",")

	X = train[:,0:21]
	Y = train[:,22]

	test = np.loadtxt("1_test.csv",delimiter = ',')
	t1 = test[:,0:21]
	t1_labels = test[:,22]

	dtclf = DecisionTreeClassifier()

	ovrc = OneVsRestClassifier(dtclf)
	ovrc.fit(X,Y)

	y_pred_ovrc = ovrc.predict(X)
	print metrics.accuracy_score(y_pred_ovrc,Y)

	test_pred_proba = ovrc.predict_proba(t1)
	test_pred = ovrc.predict(t1)

	print test_pred_proba,'\n',metrics.accuracy_score(test_pred,t1_labels)

if __name__ == '__main__':
	initialhc()