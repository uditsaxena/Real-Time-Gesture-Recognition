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
from sklearn.preprocessing import normalize as norm
from numpy.linalg import svd as svdd

np.set_printoptions(threshold=np.nan)

c = ['r','b','g','c','m','y','k','maroon', 'plum','purple','sienna','teal','violet','#b3de69','#fa8174','#00FFCC','#6d904f','olive','khaki','firebrick','crimson','navy','salmon','aliceblue','azure','beige','chocolate','coral','dimgray','fuchsia','goldenrod']

def PCA_trans(X, n_co):
	pca = PCA(n_components = n_co)
	# t = pca.fit(X)
	# print pca.explained_variance_ratio_, pca.get_params
	# matrix = np.argmax(pca.components_, axis = 1)
	# print pca.get_precision()
	# print matrix
	# print pca.explained_variance_ratio_[0:20].sum()
	# U, s, V = svdd(X, full_matrices=True)
	# print s
	return pca.fit(X)
		
def initialize(num_files, n_co_PCA, train, train_label, fold):

	GNB = []
	Model = []
	pca_per_frame = []
	
	for k in range (1,num_files):
		
		_x = train[k-1]
		_x_label = train_label[k-1]

		X = _x[fold]
		Y = _x_label[fold]
		
		# Uncomment the above lines and comment the below lines to remove PCA
		pca_X = PCA_trans(X, n_co_PCA)
		# print pca_X.components_, pca_X.explained_variance_ratio_
		# matrix = np.argsort(pca_X.components_)
		# print matrix
		pca_per_frame.append(pca_X)

		X_trans = pca_X.transform(X)
		
		nbclf = GaussianNB()
		nbclf.fit(X_trans,Y)

		y_pred_proba = nbclf.predict_proba(X_trans)
		t = np.zeros((19,19), dtype = float)
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

def get_folds(frames_to_consider, n_fold, total_instances):

	train, train_label, test, test_label = [], [], [], []
	for i in range(0, frames_to_consider):
		# print i
		data = np.loadtxt(str(i)+".txt",delimiter = ",")

		_X = data[:,0:44]
		_Y = data[:,45]

		# transferring the required number of instances
		# This is for the parts with unequal length of time series.
		X = _X[0:total_instances,:]
		Y = _Y[0:total_instances]

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
		top_index = _index_sort[:,19-top_gest:19]	

		# Stores the edge weights.
		top_index_values = _index_values[:,19-top_gest:19]

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
	A.draw('example_4'+str(top_gest)+ '.png', prog='dot')

# Alternative to "classify", trying out early class prediction.
def early_classify(num_files, trained_model, clf, pca_per_frame, test, test_label, fold, top_gest, test_record_index, min_frame, prediction_length):
	testing_sample = test[0]
	col = testing_sample[fold].shape[0]
	class_weight = np.zeros((19,col), dtype = float)
	# Here the number of the columns are the number of test records 
	
	coming_from = []
	# coming from holds the class labels held in the previous frame
	# each matrix shape is of the form: number of test records x num of top_gest
	# initially num of frames is 0.
	frame_class_ = []

	# To be used for judging the length of the longest correct series.
	convergence_matrix = np.empty([col, 1], dtype = 'bool')
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
		
		top = max_indexes[:,19-top_gest:19]	
		top_proba = proba_pred_sorted[:,19-top_gest:19]
		
		if (k == 1):
			# This means we are at the beginning of the frames
			# No previous classes yet, just go normally to the top_gest classes
			# Hence no need for coming_from, as it is empty right now.
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
			# This part of the code is just for the purpose of graphing.
		
		# This part is for printing accuracy rates along the way.
		pred_labels = np.argmax(class_weight, axis = 0)
		print k, metrics.accuracy_score(t1_labels.astype(int), pred_labels)
		# End here.

		# Uncomment the following lines, for enabling graph formation.
		if fold == 0:
			if k == 1:
				frame_class_.append(t1_labels[test_record_index])
			# Mapping the first instance
			f_i = class_weight[:,test_record_index]
			# frame_class_ = np.argmax(f_i, axis = 0), t1_labels[0].astype(int)
			# Storing the nodes, so as to change the color and map the path in the graph later on.
			frame_class_.append(np.argmax(f_i, axis = 0))

		# Taking the norm here for normalization purposes.
		# THIS is where early classification is happening.
		# normed_class_weight = norm(class_weight, axis = 0, norm = 'l1')
		
		# This 'if' condition builds the matrix for checking out how early we can classify.
		if (k >= min_frame):
			predicted_labels = np.argmax(class_weight, axis = 0)
			 
		# 	# This statement has a "LOT" going on. The reshaping is just for the stacking purpose because, 
		# 	# the shape of np.equal is of type (xxxx, ) 
		# 	# Just making it into a col vector.
			convergence_matrix = np.hstack((convergence_matrix, np.equal(t1_labels.astype(int), predicted_labels).reshape(predicted_labels.shape[0], 1)))

		
		coming_from.append(top)

	pred_labels = np.argmax(class_weight, axis = 0)

	# if (fold == 0):
	
	# c = ['r','b','g','c','m','y','k','maroon', 'plum','purple','sienna','teal','violet','#b3de69','#fa8174']
	# for length in range(1,prediction_length+1):
	# 	frame_slot, acc_slot = find_prediction_length(convergence_matrix[1:,:], length)
		
	# 	plt.plot(frame_slot, acc_slot, marker = 'o', color = c[length], linestyle = '--', label = str(length)+" as Longest running length.")
	# plt.xlabel('Number of frames')
	# plt.ylabel('Accuracy Score')
	# plt.title('Longest slot of correct gestures vs Accuracy.')
	# plt.legend()
	# plt.show()
	# # plt.savefig("output"+str(fold)+".png")
	# plt.clf()

	return metrics.accuracy_score(t1_labels.astype(int), pred_labels), frame_class_
	# return 0,frame_class_

def find_prediction_length(convergence_matrix, prediction_length):
	
	frame_slot = []
	acc_slot = []
	for i in range(convergence_matrix.shape[1]-prediction_length + 1):

		# print 'i:', i, 'to :', i+prediction_length,
		accuracy_matrix = np.all(convergence_matrix[:,i:i+prediction_length], axis = 1)

		frame_slot.append(i)

		# if (i==0):
		# 	print convergence_matrix[:,i:i+prediction_length]
		# 	print accuracy_matrix
		# 	print np.sum(accuracy_matrix)
		accuracy_score = np.sum(accuracy_matrix) * 100.0 / convergence_matrix.shape[0]
		acc_slot.append(accuracy_score)

	return frame_slot, acc_slot

def plot_frame_error(n_fold, frames_to_consider, n_co_PCA, train, train_label, test, test_label, top_gest, min_frame, prediction_length):
	
	for top_gest in range(1, top_gest):
		x_axis = []
		y_axis = []
		for frames_to_consider in range(2, frames_to_consider):
			x_axis.append(frames_to_consider)
			error = 0.0
			print "Frame: ", frames_to_consider, ':'
			for n_f in range(n_fold):
				# print 'Number of fold: ', ':', n_f,
				# print 'prediction_length', prediction_length
				trained_model, clf, pca_per_frame = initialize(frames_to_consider, n_co_PCA, train, train_label, n_f) 
				local_acc, path = early_classify(frames_to_consider, trained_model, clf, pca_per_frame, test, test_label, n_f, top_gest, 20, min_frame, prediction_length)
				error += (1.0 - local_acc)
				
				# For graphing: uncomment the following lines.
				# if (n_f == 0):
					# A is the graph returned.
					# A = make_graph(trained_model, top_gest, path)
			
			y_axis.append(100.0 * error / n_fold)
			print 100.0 * error / n_fold
		plt.plot(x_axis, y_axis, marker = 'o', color = c[top_gest], linestyle = '--', label = str(top_gest) +" nearest gestures")
	plt.xlabel('Number of frames considered')
	plt.ylabel('Error rate in %')
	plt.title('Number of frames considered vs Error Rate')
	plt.legend()
	plt.show()


def plot_PCA(n_fold, frames_to_consider, n_co_PCA, train, train_label, test, test_label, top_gest, min_frame, prediction_length):
	for top_gest in range (1, top_gest):
		x_axis = []
		y_axis = []
		for n_co_PCA in range(20, n_co_PCA):
			x_axis.append(n_co_PCA)
			error = 0.0
			print "PCA ", n_co_PCA, ':'		
			for n_f in range(n_fold):
				# print 'Number of fold: ', ':', n_f,
				# print 'prediction_length', prediction_length
				trained_model, clf, pca_per_frame = initialize(frames_to_consider, n_co_PCA, train, train_label, n_f) 
				local_acc, path = early_classify(frames_to_consider, trained_model, clf, pca_per_frame, test, test_label, n_f, top_gest, 20, min_frame, prediction_length)
				error += (1.0 - local_acc)
				
				# For graphing: uncomment the following lines.
				# if (n_f == 0):
					# A is the graph returned.
					# A = make_graph(trained_model, top_gest, path)
			y_axis.append(100.0 * error / n_fold)	
			print 100.0 * error / n_fold
		plt.plot(x_axis, y_axis, marker = 'o', color = c[top_gest], linestyle = '--', label = str(top_gest) +" nearest gestures")
	plt.xlabel('Number of features considered')
	plt.ylabel('Error rate in %')
	plt.title('Number of features considered vs Error Rate')
	plt.legend()
	plt.show()

def plot_top_gest(n_fold, frames_to_consider, n_co_PCA, train, train_label, test, test_label, top_gest, min_frame, prediction_length):
	
	x_axis = []
	y_axis = []

	for top_gest in range(1, top_gest):
		x_axis.append(top_gest)
		error = 0.0
		print "Nearest Gestures", top_gest, ':'		
		for n_f in range(n_fold):
			# print 'Number of fold: ', ':', n_f,
			# print 'prediction_length', prediction_length
			trained_model, clf, pca_per_frame = initialize(frames_to_consider, n_co_PCA, train, train_label, n_f) 
			local_acc, path = early_classify(frames_to_consider, trained_model, clf, pca_per_frame, test, test_label, n_f, top_gest, 20, min_frame, prediction_length)
			error += (1.0 - local_acc)
			
			# For graphing: uncomment the following lines.
			# if (n_f == 0):
				# A is the graph returned.
				# A = make_graph(trained_model, top_gest, path)
		y_axis.append(100.0 * error / n_fold)	
		print 100.0 * error / n_fold
	plt.plot(x_axis, y_axis, marker = 'o', color = c[top_gest], linestyle = '--')#, label = str(top_gest) +" nearest gestures")
	plt.xlabel('Number of nearest gestures')
	plt.ylabel('Error rate in %')
	plt.title('Number of nearest gestures vs Error Rate')
	plt.legend()
	plt.show()

def plot_nothing(n_fold, frames_to_consider, n_co_PCA, train, train_label, test, test_label, top_gest, min_frame, prediction_length):
	
	error = 0.0
	for n_f in range(n_fold):
		# print 'Number of fold: ', ':', n_f,
		# print 'prediction_length', prediction_length
		trained_model, clf, pca_per_frame = initialize(frames_to_consider, n_co_PCA, train, train_label, n_f) 
		local_acc, path = early_classify(frames_to_consider, trained_model, clf, pca_per_frame, test, test_label, n_f, top_gest, 20, min_frame, prediction_length)
		error += (1.0 - local_acc)
		
		# For graphing: uncomment the following lines.
		# if (n_f == 0):
			# A is the graph returned.
			# A = make_graph(trained_model, top_gest, path)
		
	print 100.0 * error / n_fold


# The main function: 
# If not plotting: 
if __name__ == '__main__':
	# c = ['r','b','g','c','m','y','k','maroon', 'plum','purple','sienna','teal','violet','#b3de69','#fa8174','#00FFCC','#6d904f','olive','khaki','firebrick','crimson','navy','salmon','aliceblue','azure','beige','chocolate','coral','dimgray','fuchsia','goldenrod']
	
	prediction_length = 5
	top_gest = 6 # the top 'x' gestures to consider.
	frames_to_consider = 125
	n_co_PCA = 44 # get this number through analysis.
	
	total_instances = 9120

	n_fold = 3
	# print "Value of K in Stratified K-fold:", n_fold

	min_frame = 7
	train, train_label, test, test_label = get_folds(frames_to_consider, n_fold, total_instances)

	plot_nothing(n_fold, frames_to_consider, n_co_PCA, train, train_label, test, test_label, top_gest, min_frame, prediction_length)

	# plot_top_gest(n_fold, frames_to_consider, n_co_PCA, train, train_label, test, test_label, top_gest, min_frame, prediction_length)

	# plot_PCA(n_fold, frames_to_consider, n_co_PCA, train, train_label, test, test_label, top_gest, min_frame, prediction_length)

	# plot_frame_error(n_fold, frames_to_consider, n_co_PCA, train, train_label, test, test_label, top_gest, min_frame, prediction_length)

	# error = 0.0
			
	# for n_f in range(n_fold):
	# 	# print 'Number of fold: ', ':', n_f,
	# 	# print 'prediction_length', prediction_length
	# 	trained_model, clf, pca_per_frame = initialize(frames_to_consider, n_co_PCA, train, train_label, n_f) 
	# 	local_acc, path = early_classify(frames_to_consider, trained_model, clf, pca_per_frame, test, test_label, n_f, top_gest, 20, min_frame, prediction_length)
	# 	error += (1.0 - local_acc)
		
	# 	# For graphing: uncomment the following lines.
	# 	# if (n_f == 0):
	# 		# A is the graph returned.
	# 		# A = make_graph(trained_model, top_gest, path)
		
	# print 100.0 * error / n_fold
	# 		x_axis.append(frames_to_consider)
	# 		error_rate.append(100.0 * error / n_fold)
	# 	plt.plot(x_axis, error_rate, marker = 'o', color = c[n_co_PCA], linestyle = '--', label = str(n_co_PCA) +" features")
	# plt.xlabel('Number of frames considered')
	# plt.ylabel('Error rate in %')
	# plt.title('Number of frames considered vs Error Rate')
	# plt.legend()
	# plt.show()