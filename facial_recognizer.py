import h5py
import sys
from tqdm import tqdm
import matplotlib.pyplot as plot

from cnn import DenseLayer
from cnn import ConvPoolLayer
from cnn import LeNet5


epochs = 100
t = 0.8

with h5py.File(sys.argv[1]) as f:
	data = f['data'][()]
	targets = f['targets'][()]

train_data = data[:int(len(data.shape[0]) * t)]
train_targets = targets[:int(len(data.shape[0]) * t)]
test_data = data[int(len(data.shape[0]) * t):]
test_targets = data[int(len(data.shape[0]) * t):]

model = LeNet5(2, 1, 0.001)

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

plot.plot(loss)
plot.savefig('losses.png')
