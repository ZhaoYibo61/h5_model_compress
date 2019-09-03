import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

# create some data
X = np.linspace(-1, 1, 4000000)
np.random.shuffle(X)  # randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (4000000,))
X_train, Y_train = X[:3200000], Y[:3200000]  # first 160 data points
X_test, Y_test = X[3200000:], Y[3200000:]  # last 40 data points
model = Sequential()
model.add(Dense(output_dim=50, input_dim=1))
model.add(Dense(output_dim=1, input_dim=1))
model.compile(loss='mse', optimizer='sgd')
for step in range(61):
    cost = model.train_on_batch(X_train, Y_train)

# save
print('test before save: ', model.predict(X_test[0:2]))
model.save('my_model1.h5')  # HDF5 file, you have to pip3 install h5py if don't have it
del model  # deletes the existing model

# load
model = load_model('my_model1.h5')
print('test after load: ', model.predict(X_test[0:2]))