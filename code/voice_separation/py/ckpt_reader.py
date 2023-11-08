# https://stackoverflow.com/questions/40118062/how-to-read-weights-saved-in-tensorflow-checkpoint-file
# https://copyfuture.com/blogs-details/202201161722057568

#from tensorflow.python.training import py_checkpoint_reader
import tensorflow as tf

file_name = "./weights/trained.ckpt"
#reader = py_checkpoint_reader.NewCheckpointReader(file_name)
reader = tf.train.NewCheckpointReader(file_name)

state_dict = {
    v: reader.get_tensor(v) for v in reader.get_variable_to_shape_map()
}

print(len(state_dict))
print(state_dict.keys())
for key in state_dict.keys():
	print(key, state_dict[key].shape)
	print(state_dict[key])