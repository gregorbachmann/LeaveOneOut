from data import CIFAR10
from kernels import NTK
from utils import mse_loss, accuracy


# Define the parameters
n_train = 2000
n_test = 10000
classes = 10
dataset = 'CIFAR'
flat = True

# Prepare the data
data = CIFAR10(n_train=n_train, n_test=n_test, classes=10, flat=flat)

# Define the model and fit it to the training data
model = NTK(depth=3)
model.fit(data.x_train, data.y_train)

# Predict on train and test set
preds_train = model.predict(data.x_test, mode='train')
preds_test = model.predict(data.x_test, mode='test')

# Calculate the statistics
train_loss = mse_loss(preds_train, data.y_train)
train_acc = accuracy(preds_train, data.y_train)
test_loss = mse_loss(preds_test, data.y_test)
test_acc = accuracy(preds_test, data.y_test)
loo_loss, loo_acc = model.leave_one_out()

print('----------Loss----------')
print('Training Loss:     ' + '%.3f' % train_loss)
print('Test Loss:         ' + '%.3f' % test_loss)
print('LOO Loss:          ' + '%.3f' % loo_loss)
print('--------Accuracy--------')
print('Training Accuracy: ' + '%.3f' % train_acc)
print('Test Accuracy:     ' + '%.3f' % test_acc)
print('LOO Accuracy:      ' + '%.3f' % loo_acc)