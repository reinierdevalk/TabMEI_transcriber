import os
from stat import *
from sys import argv

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import numpy as np
import pandas as pd

script, mdl, arg_mode, arg_path, arg_fv_ext, arg_cl_ext, arg_outp_ext, arg_param = argv

"""
This script trains the given model on the data at the given path, and then stores
both the output and the model in the same path. 

Arguments
mdl 			the model used
arg_mode 		the mode: train, test, or application
arg_path		the path where to load and store the data and model
arg_fv_ext		the extension used to load the feature vectors
arg_cl_ext 		the extension used to load the ground truth (the classes)
arg_outp_ext	the extension used to store the model output
arg_param		the parameter to tune
"""

#print(mdl)
#print(mode)
#print(path)
#print(arg_fv_ext)
#print(arg_cl_ext)
#print(arg_outp_ext)

# Set mode
train, test = False, False
if mode == 'trn':
	train = True
elif mode == 'tst':
	test = True

# Set correct extension
fv_ext = arg_fv_ext + mode + '.csv'
cl_ext = arg_cl_ext + mode + '.csv'
outp_ext = arg_outp_ext + mode + '.csv'

# Create model
m = None
if train: 
	if mdl.startswith('LR'):	
		m = LogisticRegression(C=float(param))
	elif mdl.startswith('LSVC'):
		m = LinearSVC(C=float(param))
	elif mdl.startswith('RF'):
		m = RandomForestClassifier(n_estimators=int(float(param)))
	elif mdl.startswith('kNN'):
		m = KNeighborsClassifier(n_neighbors=int(float(param)))
if test:
	m = joblib.load(path + mdl + '.pkl')

# Load data
X = np.loadtxt((path + fv_ext), delimiter=",")
if train:
	y = np.loadtxt((path + cl_ext), delimiter=",")
##	X_list = [float(s.strip()) for s in arg_fv_ext.split(',')]
##	X = np.array(X_list)
###	for d in X:
###		print("%.20f" % d)
##	# Reshape data in application case to avoid deprecation warning

# Fit model and get results
if train:
	m.fit(X, y)

classes = m.predict(X) # ndarray
max_num_voices = 5
num_ex = len(classes)

if not mdl.endswith('CL'):
	probs = m.predict_proba(X) # ndarray
	# Complete probs with colums of zeroes
	num_cols_to_add = max_num_voices - len(probs[0]) 
#	num_ex = len(probs) 
	z = np.zeros((num_ex, num_cols_to_add), dtype=probs.dtype)
	probs = np.append(probs, z, axis=1)
else: #if mdl.endswith('CL'):
	probs = np.zeros((num_ex, max_num_voices))
	for i in range (0, num_ex):
		probs[i][int(classes[i])] = 1.0 

#set_printoptions(precision=20)
# Save output and, in training mode, the model 
df_probs = pd.DataFrame(probs)
df_probs.to_csv(path + outp_ext, header=False, index=False)
#df_classes = pd.DataFrame(classes)
#df_classes.to_csv(path + 'outp_cl_' + mode + '.csv', header=False, index=False)

if train:
	joblib.dump(m, path + mdl + '.pkl')

# Print stuff
verbose=False
if verbose:
#	lab = y
	corr = 0
	for i in range(0, len(classes)):
#		print('target = ' + str(lab[i]) + ' --> pred = ' + str(classes[i]) + ', probs = ' + str(probs[i]))
		if classes[i] == lab[i]:
			corr += 1 	
		max_val = np.argmax(probs[i]) 
		if max_val != classes[i]:
			print('warning: class and prob are not the same at index ' + i)
	acc = corr / len(X)
	print(acc)

if train:
	print(m.score(X, y))