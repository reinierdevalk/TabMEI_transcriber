import os
from sys import argv
import tensorflow as tf
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

train = 0
test = 1
appl = 2

#seed=0 # seed=0 used for all experiments ISMIR 2018 paper 

# when used as a script
if len(argv) > 1:
	script, mdl, arg_mode, arg_path, arg_exts, arg_params, arg_stored_weights = argv

	"""
	This script trains the given model on the data at the given path, and then stores
	both the output and the weights in the same path. 

	Arguments
	mdl				the model used
	arg_mode 			the mode: train, test, or application
	arg_path			the path where to load and store the data and model
	arg_exts			the extension used for the feature vectors, labels, and model outputs
	arg_param			the parameters and hyperparameters to tune
	arg_stored_weights	whether or not to initialise the network with the stored initial weights
	"""

	print('mdl', mdl)
	print('arg_mode', arg_mode)
	print('arg_path', arg_path)
	print('arg_exts', arg_exts)
	print('arg_params', arg_params)
	print('arg_stored_weights', arg_stored_weights)
	print()

	nums_tst = []
	dens_tst = []

	if arg_mode == 'trn':
		mode = train
	elif arg_mode == 'tst':
		mode = test
	elif arg_mode == 'app':
		mode = appl

	exts = [s.strip() for s in arg_exts.split(',')]
	fv_ext = exts[0] + arg_mode + '.csv'
	lbl_ext = exts[1] + arg_mode + '.csv'
	out_ext = exts[2] + arg_mode + '.csv'

	use_stored_weights = False
	if arg_stored_weights.casefold() == 'true':
		use_stored_weights = True

	fold_path = arg_path
#	num_features = len(genfromtxt(fold_path + fv_ext, delimiter=',')[0])
#	num_classes = len(genfromtxt(fold_path + lbl_ext, delimiter=',')[0])

	# parameters and hyperparameterss
	params = [s.strip() for s in arg_params.split(',')]
	param_dict = {}
	for item in params:
		nameVal = item.strip().split('=')
		if '.' in nameVal[1]:
			param_dict[nameVal[0]] = float(nameVal[1])
		else:
			param_dict[nameVal[0]] = int(nameVal[1])

	num_HL =  param_dict['hidden layers'] #int(float(arg_param))
	IL_size = param_dict['input layer size']
	HL_size = param_dict['hidden layer size']
	OL_size = param_dict['output layer size']
	lrn_rate = param_dict['learning rate']
	kp = param_dict['keep probability']
	epochs = param_dict['epochs'] # one epoch is one fwd-bwd propagation
#	num_nodes_HL = [num_features, num_features, num_features, num_features]                                  
#	layer_sizes = [num_features]
	layer_sizes = [IL_size]
	for i in range(num_HL):
#		layer_sizes.append(num_nodes_HL[i])
		layer_sizes.append(HL_size)
#	layer_sizes.append(num_classes)
	layer_sizes.append(OL_size)
	
	print('num_HL', num_HL)
	print('IL_size', IL_size)
	print('HL_size', HL_size)
	print('OL_size', OL_size)
	print('lrn_rate', lrn_rate)
	print('kp', kp)
	print('epochs', epochs)
	print(layer_sizes)

# when used as a module
else:
	script = argv
	mode = appl
	fv_ext = 'x-app.csv'
	lbl_ext = 'y-app.csv'
	out_ext = 'out-app.csv'


tf.reset_default_graph()

if mode == train:
	x_train = genfromtxt(fold_path + fv_ext, delimiter=',')
	y_train = genfromtxt(fold_path + lbl_ext, delimiter=',')
	x_val = genfromtxt(fold_path + exts[0] + 'vld.csv', delimiter=',')
	y_val = genfromtxt(fold_path + exts[1] + 'vld.csv', delimiter=',')
	batch_size = len(x_train)  
elif mode == test:
	x_test  = genfromtxt(fold_path + fv_ext, delimiter=',')
#	print(x_test.shape)
	y_test  = genfromtxt(fold_path + lbl_ext, delimiter=',')
	use_stored_weights = True
#elif mode == appl:
#	x_appl  = genfromtxt(fold_path + fv_ext, delimiter=',')
#	y_appl  = genfromtxt(fold_path + lbl_ext, delimiter=',')
#	use_stored_weights = True

if mode == train or mode == test:
	x = tf.placeholder('float', [None, IL_size])
	y = tf.placeholder('float')
	keep_prob = tf.placeholder('float')


def create_neural_network(layer_sizes, use_stored_weights, mode, fold_path):
	"""
	Creates the neural network.

	Arguments:
		layer_sizes				the sizes of the input, hidden, and output layers
		use_stored_weights		whether or not to initialise the network with stored weights
		mode 					the evaluation mode: train, test, or application
	"""

	print('create_neural_network called')
	# hidden and output layers
	num_layers = len(layer_sizes) - 1
	weights = {}
	biases = {}

	print('creating a DNN with', str(num_layers-1), 'hidden layers of size', str(layer_sizes[1:len(layer_sizes)-1]))

	# initialise the weights
	# (a) create new weights and biases
	if not use_stored_weights:
		print('use_stored_weights =', use_stored_weights)
		for i in range(num_layers):            
			# layer l has dimensions (|l-1|, |l|) for weights and (|l|) for biases
			w_name = 'W' + str(i+1)
			weights[w_name] = tf.get_variable(w_name, [layer_sizes[i], layer_sizes[i+1]], 
				initializer = tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			b_name = 'b' + str(i+1)
			biases[b_name] = tf.get_variable(b_name, [layer_sizes[i+1]], 
				initializer = tf.zeros_initializer(), dtype=tf.float32)

		# save weights and biases
		saver = tf.train.Saver()
		#with tf.Session() as sess:
		# Initialise all existing global variables 
		sess.run(tf.global_variables_initializer())
		save_path = saver.save(sess, fold_path + 'weights/' + 'init.ckpt')
#		print('W1', sess.run(weights['W1']))
		# END OF SESSION    

	# (b) restore existing weights and biases
	else:
		print('use_stored_weights = ', use_stored_weights)
		for i in range(num_layers):
			# prepare variable
			w_name = 'W' + str(i+1)
			b_name = 'b' + str(i+1)
			weights[w_name] = tf.get_variable(w_name, [layer_sizes[i], layer_sizes[i+1]], 
				initializer = tf.zeros_initializer(), dtype=tf.float32)
			biases[b_name] = tf.get_variable(b_name, [layer_sizes[i+1]], 
				initializer = tf.zeros_initializer(), dtype=tf.float32)

		saver = tf.train.Saver()
		#with tf.Session() as sess:
		# Initialise all existing global variables 
		sess.run(tf.global_variables_initializer())
		if mode == train:
			saver.restore(sess, fold_path + 'weights/' + 'init.ckpt')            
		elif mode == test or mode == appl:
			saver.restore(sess, fold_path + 'weights/' + 'trained.ckpt')            
#			print('W1', sess.run(weights['W1']))

		# END OF SESSION

	wb = {'weights': weights, 'biases': biases}
	return wb
# fed	

#	# calculate linear and relu outputs for the hidden layers
#	a_prev = data
#	for i in range(num_layers-1):  
##		with tf.Session() as sess:
##		print('W1', sess.run(weights['W1']))
#		z = tf.add(tf.matmul(a_prev, weights['W' + str(i+1)]), biases['b' + str(i+1)])
#		a = tf.nn.relu(z)
#		a_r = tf.nn.dropout(a, keep_prob, seed=0)
#		a_prev = a_r
#	# calculate linear output for the output layer
#	z_o = tf.add(tf.matmul(a_prev, weights['W' + str(num_layers)]), biases['b' + str(num_layers)])
#
#	return z_o


def evaluate_neural_network(data, keep_prob, num_layers, weights, biases):
	
	"""
	Evaluates the neural network.

	Arguments:
		data					the features
		num_layers				the number of hidden and output layers
		weights 				the weight
		biases 					the biases
	"""

#	print(sess.run(weights['W1']))
#	print(sess.run(biases['b1']))

	# calculate linear and relu outputs for the hidden layers
	a_prev = data
	for i in range(num_layers-1):
#		with tf.Session() as sess:
#		print('W1', sess.run(weights['W1']))  
		z = tf.add(tf.matmul(a_prev, weights['W' + str(i+1)]), biases['b' + str(i+1)])
		a = tf.nn.relu(z)
#		a_r = tf.nn.dropout(a, keep_prob, seed=seed) # VANDAAG
		a_r = tf.nn.dropout(a, keep_prob, seed=0) # VANDAAG 
		a_prev = a_r
	# calculate linear output for the output layer
	z_o = tf.add(tf.matmul(a_prev, weights['W' + str(num_layers)]), biases['b' + str(num_layers)])

	return z_o
# fed


def run_neural_network(x, keep_prob, lrn_rate, kp, epochs, layer_sizes, use_stored_weights, mode, fold_path, *args):
	"""
	Runs the neural network.

	Arguments:
		x						the features
		layer_sizes				the sizes of the input, hidden, and output layers
		use_stored_weights		whether or not to initialise the network with stored weights
		mode 					the evaluation mode: train, test, or application
		*args 					the weights and biases dictionary (only when mode is application)
	"""

	print('run_neural_network called in mode',  mode)
#	prediction = create_neural_network(x, layer_sizes, use_stored_weights, mode) # train and test
	
	# train, test, application
	weights_biases = {}
	if mode == train or mode == test:
		weights_biases = create_neural_network(layer_sizes, use_stored_weights, mode, fold_path)
	elif mode == appl:
		weights_biases = args[0]

	prediction = evaluate_neural_network(x, keep_prob, len(layer_sizes) - 1, weights_biases['weights'], weights_biases['biases'])

	softm = tf.nn.softmax(prediction)
	pred_class = tf.argmax(softm)

	# train and test
	if mode == train or mode == test:

		if mode == train:
			cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
			optimizer = tf.train.AdamOptimizer(learning_rate=lrn_rate).minimize(cost)
		
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		# initialise all global variables that have not been initialised yet (e.g., variables for Adam)
#		with tf.Session() as sess:
#		sess.run(tf.global_variables_initializer())
		# see Salvador Dali's answer at 
		# https://stackoverflow.com/questions/35164529/in-tensorflow-is-there-any-way-to-just-initialize-uninitialised-variables
		global_vars = tf.global_variables()
		is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
		not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
#		print('uninitialised variables:', [str(i.name) for i in not_initialized_vars])
		if len(not_initialized_vars):
			sess.run(tf.variables_initializer(not_initialized_vars))
		print('uninitialised variables:', sess.run(tf.report_uninitialized_variables()))

	# train
	if mode == train:
#		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
#		optimizer = tf.train.AdamOptimizer(learning_rate=lrn_rate).minimize(cost)
		costs = []
		accs_tr = []
		accs_val = []
		best_acc = 0.0
		for epoch in range(epochs):
			epoch_loss = 0
#			for _ in range(int(mnist.train.num_examples/batch_size)):
			for _ in range(int(len(x_train)/batch_size)):				
#				new_sets = make_validation_set(x_train, y_train, 20)
#				x_train_new = new_sets['x_tr_new']
#				y_train_new = new_sets['y_tr_new']
#				x_val = new_sets['x_val']
#				y_val = new_sets['y_val']
				# epoch_x and epoch_y are numpy.ndarray
#				epoch_x, epoch_y = mnist.train.next_batch(batch_size) #iterates through the dataset
				epoch_x, epoch_y = x_train, y_train
#				epoch_x, epoch_y = x_train_new, y_train_new
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y, keep_prob: kp})
				epoch_loss += c

#			if epoch % 40 == 0:
#				print('epoch', epoch, 'completed out of', epochs, '; loss =', epoch_loss)
			if epoch % 10 == 0:
				acc_tr = accuracy.eval({x:x_train, y:y_train, keep_prob: kp}) ## HIERO
				acc_val = accuracy.eval({x:x_val, y:y_val, keep_prob: 1.0})
#				corrrr = correct.eval({x:x_val, y:y_val, keep_prob: 1.0});
				
				costs.append(epoch_loss)
				accs_tr.append(acc_tr) ## HIERO
				accs_val.append(acc_val) ## HIERO

				dothis = False
				if dothis:
					softmaxes_val = sess.run([softm, pred_class], feed_dict={x: x_val, keep_prob: 1.0})[0]
					incorr = 0
					for i in range(len(softmaxes_val)):
						curr_sm = softmaxes_val[i]
						curr_lbl = y_val[i]
						if np.argmax(curr_sm) != np.argmax(curr_lbl):
#						print(i)
#						print(curr_sm)
#						print('pred class = ' + str(np.argmax(curr_sm)))
#						print(curr_lbl)
#						print('corr class = ' + str(np.argmax(curr_lbl)))
							incorr += 1

#					print('incorr: ', incorr)
					num_ex = len(x_val)
#					acc_val = (num_ex-incorr)/num_ex

#					print('volgens mij, acc val = ' + str((num_ex - incorr)/num_ex)) #, 'volgens tf, acc tr =' + str(acc_tr))
			
				saver = tf.train.Saver()
				if acc_val > best_acc:
#					print('best acc val is now', acc_val, '(epoch =', epoch, ')')#), 'best acc trn is now', acc_tr)
					best_acc = acc_val
					save_path = saver.save(sess, fold_path + 'weights/' + 'trained.ckpt')
					# save the model output
					# see https://stackoverflow.com/questions/6081008/dump-a-numpy-array-into-a-csv-file
					softmaxes_trn = sess.run([softm, pred_class], feed_dict={x: x_train, keep_prob: kp})[0]
					np.savetxt(fold_path + out_ext, softmaxes_trn, delimiter=",")
#					np.savetxt(fold_path + 'best_epoch.txt', 'highest accuracy on the validation set (' + str(best_acc) + ') in epoch ' + str(epoch), delimiter="", fmt="%s")			
					with open(fold_path + 'best_epoch.txt', 'w') as text_file:
						text_file.write('highest accuracy on the validation set (' + str(best_acc) + ') in epoch ' + str(epoch))

		print(accs_tr) ## HIERO
		print(accs_val) ## HIERO


		# save the weights
#		saver = tf.train.Saver()    
#		save_path = saver.save(sess, fold_path + 'weights/' + 'trained.ckpt')

#		# save the model output
#		# see https://stackoverflow.com/questions/6081008/dump-a-numpy-array-into-a-csv-file
#		softmaxes_trn = sess.run([softm, pred_class], feed_dict={x: x_train_new, keep_prob: kp})[0]
#		np.savetxt(fold_path + 'out_trn.csv', softmaxes_trn, delimiter=",")

		# plot the cost
#		print('accuracy train:', accuracy.eval({x:x_train, y:y_train, keep_prob: kp}))
#		plt.plot(np.squeeze(costs))
#		plt.ylabel('cost')
#		plt.xlabel('epochs (/10)')
#		plt.title('learning rate =' + str(lrn_rate))
#		plt.show()

		# plot the tr and val accuracy
		plotOrNot = True
		if plotOrNot:
			plt.plot(np.squeeze(accs_tr))
			plt.plot(np.squeeze(accs_val))
			plt.ylabel('acc')
			plt.xlabel('epochs (/10)')
			ax = plt.subplot(111)
			ax.set_prop_cycle('color', ['red', 'green'])
#			plt.gca().set_prop_cycle(['red', 'green'])
			plt.title('accuracy on training and validation set')
			plt.legend(['tr', 'val'], loc='lower right')
#			plt.show()
			plt.savefig(fold_path + 'trn_and_val_acc.png')

	# test
	if mode == test:
		softmaxes_tst = sess.run([softm, pred_class], feed_dict={x: x_test, keep_prob: 1.0})[0]
		np.savetxt(fold_path + out_ext, softmaxes_tst, delimiter=",")
 
		dothis = False
		if dothis:
			incorr = 0
			for i in range(len(softmaxes_tst)):
				curr_sm = softmaxes_tst[i]
				curr_lbl = y_test[i]
				if np.argmax(curr_sm) != np.argmax(curr_lbl):
#				print(i)
#				print(curr_sm)
#				print('pred class = ' + str(np.argmax(curr_sm)))
#				print(curr_lbl)
#				print('corr class = ' + str(np.argmax(curr_lbl)))
					incorr += 1

			print('incorr: ', incorr)
			num_ex = len(x_test)
			print('acc = ' + str(num_ex - incorr) + '/' + str(num_ex) + ' = ' + str((num_ex - incorr)/num_ex))

			nums_tst.append(num_ex - incorr)
			dens_tst.append(num_ex)

			print('accuracy test :', accuracy.eval({x:x_test, y:y_test, keep_prob: 1.0})) 
#		output = sess.run(prediction,feed_dict={x: x_test})
	# END OF SESSION

	if mode == appl:
		# Features loaded from file

		x_appl = genfromtxt(fold_path + fv_ext, delimiter=',')
		# Features given as argument		
#		list_from_string = [float(s.strip()) for s in feature_vector.split(',')]
#		x_appl = np.array(list_from_string)
		# reshape necessary to get required shape (1, 33)
		x_appl = x_appl.reshape(1, -1)
		softmaxes_app = sess.run([softm, pred_class], feed_dict={x: x_appl, keep_prob: 1.0})[0]
		np.savetxt(fold_path + out_ext, softmaxes_app, delimiter=",")

#	this is very slow: https://stackoverflow.com/questions/49041001/why-is-this-tensorflow-code-so-slow 
#	output = sess.run(prediction,feed_dict={x: x_test})
#	incorr = 0
#	for i in range(len(x_test)):
#		print(i) 
#		softm = sess.run(tf.nn.softmax(output[i]))
#		pred_class = sess.run(tf.argmax(softm))
#		act_class = sess.run(tf.argmax(y_test[i]))            
#		if pred_class != act_class:
#			incorr += 1    
#	print('acc = ' + str(incorr) + '/' + str(len(x_test)) + ' = ' + str(incorr/len(x_test)))

# fed


def make_validation_set(x_tr, y_tr, perc):
	"""
	Takes every nth exmaple from the training set and puts it into a validation set. n is determined by the given 
	percentage: n = 100/perc  

	Arguments:
		x_tr					the features
		y_tr					the labels
		perc					the percentage of training examples that go into the validation set
	"""

	x_val, y_val, x_tr_new, y_tr_new = [], [], [], []
	n = 100/perc
	for i in range(len(x_tr)):
		if i % n == 0:
			x_val.append(x_tr[i])
			y_val.append(y_tr[i])
		else:
			x_tr_new.append(x_tr[i])
			y_tr_new.append(y_tr[i])
 
	arrs = {'x_val': np.array(x_val), 
			'y_val': np.array(y_val), 
			'x_tr_new': np.array(x_tr_new), 
			'y_tr_new': np.array(y_tr_new)}
	return arrs


# interactive session needed to be able to share variables between functions; 
# 'with tf.Session() as sess:' does not work
# see https://www.tensorflow.org/api_docs/python/tf/InteractiveSession
# example of passing around sessions at  
# https://stackoverflow.com/questions/44660572/function-scopes-with-tensorflow-sessions
sess = tf.InteractiveSession() 
# set seed to ensure deterministic dropout
# see https://stackoverflow.com/questions/49175704/tensorflow-reproducing-results-when-using-dropout
#tf.set_random_seed(seed) # VANDAAG
tf.set_random_seed(0) # VANDAAG
if mode == train or mode == test:
	run_neural_network(x, keep_prob, lrn_rate, kp, epochs, layer_sizes, use_stored_weights, mode, fold_path)
#elif mode == appl:
#	weights_biases = create_neural_network(layer_sizes, use_stored_weights, mode, fold_path)
#	# https://pythontips.com/2013/08/04/args-and-kwargs-in-python-explained/
#	run_neural_network(x, keep_prob, lrn_rate, kp, epochs, layer_sizes, use_stored_weights, mode, fold_path, weights_biases)
sess.close()

#if mode == test:
#	print(nums_tst)
#	print(dens_tst)
#	print(str(sum(nums_tst)/sum(dens_tst)))

