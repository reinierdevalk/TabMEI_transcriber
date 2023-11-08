import os
from stat import *
from sys import argv

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import numpy as np
import pandas as pd

script, mdl, mode, path, arg_fv_ext, arg_cl_ext, arg_outp_ext = argv

"""
This script trains the given model on the data at the given path, and then stores
both the output and the model in the same path. 

Arguments
mdl 			the model used
mode 			the mode: train, test, or application
path			the path where to load and store the data and model
arg_fv_ext		the extension used to load the feature vectors, or, in the appl case,
                the full feature vector as a string
arg_cl_ext 		the extension used to load the ground truth (the classes)
arg_outp_ext	the extension used to store the model output
"""

#print(mdl)
#print(mode)
#print(path)
#print(arg_fv_ext)
#print(arg_cl_ext)
#print(arg_outp_ext)

# Set mode
train, test, appl = False, False, False
if mode == 'train':
	train = True
elif mode == 'test':
	test = True
elif mode == 'appl':	 
	appl = True

### Set correct extension
##if not appl:
fv_ext = arg_fv_ext + mode + '.csv'
cl_ext = arg_cl_ext + mode + '.csv'
outp_ext = arg_outp_ext + mode + '.csv'

# Create model
m = None
if train: 
	if mdl == 'LR':	
		m = LogisticRegression()
if test or appl:
	m = joblib.load(path + mdl + '.pkl')

# Load data
if train:
	X = np.loadtxt((path + fv_ext), delimiter=",")
	y = np.loadtxt((path + cl_ext), delimiter=",")
if test:
	X = np.loadtxt((path + fv_ext), delimiter=",")
if appl:
#	X = np.loadtxt((path + fv_ext), delimiter=",")
	X_list = [float(s.strip()) for s in arg_fv_ext.split(',')]
	X = np.array(X_list)
	for d in X:
		print("%.20f" % d)
	# Reshape data in application case to avoid deprecation warning
	X = X.reshape(1, -1)

# Fit model and get results
if train:
	m.fit(X, y)

classes = m.predict(X) # ndarray
probs = m.predict_proba(X) # ndarray
# Complete probs with columns of zeroes 
max_num_voices = 5
num_cols_to_add = max_num_voices - len(probs[0]) 
num_ex = len(probs)
z = np.zeros((num_ex, num_cols_to_add), dtype=probs.dtype)
probs = np.append(probs, z, axis=1)

np.set_printoptions(precision=20)

##if appl:
##	output = ','.join([str(p) for p in probs[0]])
##	print('model output = ' + output)

# Save output and, in training mode, the model 
##if train or test:
df_probs = pd.DataFrame(probs)
df_probs.to_csv(path + outp_ext, header=False, index=False)
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