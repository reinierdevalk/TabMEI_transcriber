from pandas import DataFrame, concat, read_csv
from keras.models import model_from_json
from sys import argv

script, path = argv

# load best model
# load json and create model
json_file = open(path + '/' + 'best_model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(path + '/' + 'best_model/model.h5')
print('Loaded model from disk')

loaded_model.compile(loss='mean_squared_error', optimizer='adam')
model = loaded_model

# test_X contains, per note, a list of V sequences, one per voice, containing 
# a list of notes or equalling None (in case their last note is sustained)
# each of these must be reformatted
test_X = read_csv(path + '/' + 'seqs_tst.csv', header=0)
features = test_X.values

pred = model.predict(test_X, batch_size=batch_size)