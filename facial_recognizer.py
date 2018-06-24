import h5py
import sys
import numpy
from tqdm import tqdm
import matplotlib.pyplot as plot

from cnn import LeNet5

epochs = 10
t = 0.8

with h5py.File(sys.argv[1], 'r') as f:
    data = f['data'][()]
    targets = f['targets'][()]

data = data / 255.0

train_data = data[:int(data.shape[0] * t)]
train_targets = targets[:int(data.shape[0] * t)]
test_data = data[int(data.shape[0] * t):]
test_targets = targets[int(data.shape[0] * t):]

model = LeNet5(input_size=(48, 48, 1), classes=2)

losses = []
prev_loss = numpy.inf
prev_model = model
for epoch in tqdm(range(epochs), desc='Epoch'):
    model.fit(train_data, train_targets)
    if epoch % 5 == 0:
        loss = model.loss(test_data, test_targets)
        losses.append(loss)
        if loss > prev_loss:
            break
        prev_loss = loss

plot.plot(losses)
plot.savefig('losses.png')
