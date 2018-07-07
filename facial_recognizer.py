import h5py
import sys
import numpy
from tqdm import tqdm
import matplotlib.pyplot as plot

import cnn_tf
import cnn_keras
import regression

epochs = 10
t = 0.8
test_frequency = 1
min_improvement = 0.0001

with h5py.File(sys.argv[1], 'r') as f:
    data = f['data'][()]
    targets = f['targets'][()]

data = data / 255.0

train_data = data[:int(data.shape[0] * t)]
train_targets = targets[:int(data.shape[0] * t)]
test_data = data[int(data.shape[0] * t):]
test_targets = targets[int(data.shape[0] * t):]

batch_size = 128

model = cnn_tf.LeNet5(input_size=(28, 28, 1), classes=10)


losses = []
accuracies = []
prev_loss = numpy.inf
prev_model = model
loss = model.loss(test_data, test_targets)
accuracy = model.accuracy(test_data, test_targets)
losses.append(loss)
accuracies.append(accuracy)
for epoch in tqdm(range(epochs), desc='Epoch'):
    model.fit(train_data, train_targets, batch_size=batch_size)
    if epoch % test_frequency == 0:
        loss = model.loss(test_data, test_targets)
        accuracy = model.accuracy(test_data, test_targets)
        losses.append(loss)
        accuracies.append(accuracy)
        if (prev_loss / loss) - 1 < min_improvement:
            break
        prev_loss = loss

#plot.plot(losses)
#plot.savefig('losses.png')

plot.plot(accuracies)
plot.savefig('accuracies.png')
