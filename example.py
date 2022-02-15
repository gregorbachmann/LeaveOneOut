from kernels import NTK
from utils import mse_loss, accuracy, get_dataset


# Define the parameters
n_train = 2000
n_test = 10000
classes = 10
dataset = 'CIFAR'
flat = True

# Prepare the data
data = get_dataset(name=dataset, n_train=n_train, n_test=n_test, classes=classes, flat=flat, download=False)

# Define the model and fit it to the training data
model = NTK(depth=5)
model.fit(data.x_train, data.y_train)

# Predict on train and test set
preds_train = model.predict(data.x_train, mode='train')
preds_test = model.predict(data.x_test, mode='test')

# Calculate the train, test and leave-one-out statistics
train_loss = mse_loss(preds_train, data.y_train)
train_acc = accuracy(preds_train, data.y_train)
test_loss = mse_loss(preds_test, data.y_test)
test_acc = accuracy(preds_test, data.y_test)
loo_loss, loo_acc = model.leave_one_out()

print('-----------Loss-----------')
print('Training Loss:     ' + '%.4f' % train_loss)
print('Test Loss:         ' + '%.4f' % test_loss)
print('LOO Loss:          ' + '%.4f' % loo_loss)
print('---------Accuracy---------')
print('Training Accuracy: ' + '%.4f' % train_acc)
print('Test Accuracy:     ' + '%.4f' % test_acc)
print('LOO Accuracy:      ' + '%.4f' % loo_acc)