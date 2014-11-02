# This is different than the file: nbclass.py as 
# the way the model is constructed at each time frame is
# different.
import sys
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import cross_validation as cv
import matplotlib.colors as colors
import networkx as nx


np.set_printoptions(threshold=np.nan)

def PCA_trans(X, n_co):
	pca = PCA(n_components = n_co)
	return pca.fit(X)
		
def initialize(num_files, n_co_PCA, train, train_label, fold):

	GNB = []
	Model = []
	pca_per_frame = []
	
	for k in range (1,num_files):
		# train = np.loadtxt(str(k)+".csv",delimiter = ",")

		# X = train[:,0:21]
		# Y = train[:,22]
		
		_x = train[k-1]
		_x_label = train_label[k-1]

		X = _x[fold]
		Y = _x_label[fold]

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

def classify(num_files, trained_model, clf, pca_per_frame, test, test_label, fold, top_gest, test_record_index):
	testing_sample = test[0]
	col = testing_sample[fold].shape[0]
	class_weight = np.zeros((95,col), dtype = float)
	# Here the number of the columns are the number of test records 
	
	coming_from = []
	# coming from holds the class labels held in the previous frame
	# each matrix shape is of the form: number of test records x num of top_gest
	# initially num of frames is 0.
	frame_class_ = []
	for k in range(1,num_files):
		n_gest = trained_model[k-1]
		# n_gest is the matrix of the corresponding time-frame
		# which holds the class predictions for the time-frame.
		
		_te = test[k-1]
		_te_label = test_label[k-1]

		t1 = _te[fold]
		t1_labels = _te_label[fold]

		
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
		if fold == 0:
			if k == 1:
				frame_class_.append(t1_labels[test_record_index])
			# Mapping the first instance
			f_i = class_weight[:,test_record_index]
			# frame_class_ = np.argmax(f_i, axis = 0), t1_labels[0].astype(int)
			# Storing the nodes, so as to change the color and map the path in the graph later on.
			frame_class_.append(np.argmax(f_i, axis = 0))

		coming_from.append(top)

	pred_labels = np.argmax(class_weight, axis = 0)

	return metrics.accuracy_score(t1_labels.astype(int), pred_labels), frame_class_

def get_folds(frames_to_consider, n_fold):

	train, train_label, test, test_label = [], [], [], []
	for i in range(1, frames_to_consider):
		data = np.loadtxt(str(i)+".csv",delimiter = ",")

		X = data[:,0:21]
		Y = data[:,22]

		skf = cv.StratifiedKFold(Y, n_fold)
		
		tr, tr_label, te, te_label = [], [], [], []

		for train_index, test_index in skf:
			tr.append(X[train_index])
			
			tr_label.append(Y[train_index])
			te.append(X[test_index])
			
			te_label.append(Y[test_index])

		train.append(tr)
		train_label.append(tr_label)
		test.append(te)
		test_label.append(te_label)

	return train, train_label, test, test_label

def make_graph(model, top_gest, path):
	print path
	label = path[0].astype(int)
	path = path[1:]
	# Utility code to deal and parse path
	j = 0
	for i in path:
		path[j] = str(i)+'.'+str(j)
		j+=1
	# path is now of the form '0.0' - '0.23' 
	# 
	
	f = []
	f_edge_l = []
	for i in range(len(model)):
		# Getting the top_gest col so as to build the graph.
		_index_sort = model[i].argsort(axis = 1)
		_index_values = np.sort(model[i],axis = 1)
		
		# Store the edge endpoints.
		top_index = _index_sort[:,95-top_gest:95]	

		# Stores the edge weights.
		top_index_values = _index_values[:,95-top_gest:95]

		# if i == 0:
		# 	print top_index,top_index_values
		# f = []
		if i == 0:
			l = []
			for j in range(model[0].shape[0]):
					a = str(j) + '.' + str(i)
					l.append(a)
		f.append(l)

		l=[]
		for j in range(model[0].shape[0]):
			b = str(j) + '.' + str(i+1)
			l.append(b)
		f.append(l)

		# f_edge_l = []
		for j in range(top_index.shape[0]):
			frm = str(j) + '.' + str(i)
			for k in range(top_gest):
				to = str(top_index[j,k]) + '.' + str(i+1)
				f_to_l = (frm, to)
				f_edge_l.append(f_to_l)
		# if i == 0:
	# Now making the graph
	G = nx.DiGraph()
	for rank_of_nodes in f:
	    G.add_nodes_from(rank_of_nodes)
	G.nodes(data=True)
	# print len(G.node)
	
	G.add_edges_from(f_edge_l)
	A = nx.to_agraph(G)
	A.graph_attr.update(rankdir='LR')  # change direction of the layout
	for rank_of_nodes in f: #ranked_node_names:
	    A.add_subgraph(rank_of_nodes, rank='same')
	# For the true label:
	for _node in A.nodes():
		if _node.startswith(str(label)):
			n = A.get_node(_node)
			n.attr['color'] = 'blue'
			n.attr['style'] = 'filled'
	# draw
	for _node in A.nodes():
		if _node in path or _node == str(label)+'.24':
			n = A.get_node(_node)
			if n.attr['color'] == 'blue':
				n.attr['color'] = 'green'
			else:
				n.attr['color'] = 'red'
			n.attr['style'] = 'filled'

	# for j in range(len(path)):
	# 	if j < len(path) - 1:
	# 		if (A.has_edge(path[j], path[j+1])):
	# 			e = A.get_edge(path[j], path[j+1])
	# 			e.attr['color'] = 'red'
	A.draw('example'+str(top_gest)+ '.png', prog='dot')


# The main function: 
# If not plotting: 
if __name__ == '__main__':
	# c = ['r','b','g','c','m','y','k','maroon', 'plum','purple','sienna','teal','violet','#b3de69','#fa8174','#00FFCC','#6d904f','olive','khaki','firebrick','crimson','navy','salmon','aliceblue','azure','beige','chocolate','coral','dimgray','fuchsia','goldenrod']
	
	top_gest = 3 # the top 'x' gestures to consider.
	frames_to_consider = 25
	n_co_PCA = 18 # got this number through analysis.
	
	n_fold = 5
	# print "Value of K in Stratified K-fold:", n_fold

	train, train_label, test, test_label = get_folds(frames_to_consider, n_fold)
	error = 0.0
	for n_f in range(n_fold):
		# print n_f, ':'
		trained_model, clf, pca_per_frame = initialize(frames_to_consider, n_co_PCA, train, train_label, n_f) 
		local_error, path = classify(frames_to_consider, trained_model, clf, pca_per_frame, test, test_label, n_f, top_gest, 336)
		error += (1.0 - local_error)
		if (n_f == 0):
			# A is the graph returned.
			A = make_graph(trained_model, top_gest, path)
	print 100.0 * error / n_fold
	
	# plt.plot(t_g, error_rate, marker = 'o', color = c[1], linestyle = '--', label = str(top_gest)+" Nearest Gestures")
	# plt.xlabel('Number of nearest gestures')
	# plt.ylabel('Error rate in %')
	# plt.title('Number of nearest gestures vs Error Rate')
	# # plt.legend()
	# plt.show()