import numpy as np
import pickle
import time
from random import uniform
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


def relu(x):
    return np.maximum(0, x)


def drelu(inputs, dinputs):
    dinputs[inputs <= 0] = 0
    return dinputs


def optimize(model, grad_buffer, lr):
    """ Vanilla Stochastic Gradient Descent """
    for k, v in model.items():
        model[k] -= lr * grad_buffer[k]


def init_grad_buffer(model):
    grad_buffer = {}
    for k, v in model.items():
        grad_buffer[f'{k}'] = np.zeros_like(v)
    return grad_buffer


def forward(inputs):
    y1 = np.dot(inputs, model['W1']) + model['Bias_W1']
    z1 = relu(y1)
    y2 = np.dot(z1, model['W2']) + model['Bias_W2']
    z2 = relu(y2)
    values = np.dot(z2, model['W3']) + model['Bias_W3']

    return {'inputs': inputs, 'y1': y1, 'z1': z1, 'y2': y2, 'z2': z2}, values


def loss_fun(values, targets):
    loss = np.mean((targets - values) ** 2)
    return loss


def backward(values, hidden, targets):
    samples = len(values)
    outputs = len(values[0])

    dinputs = values.copy()

    # Calculate gradient
    dinputs = -2 * (targets - dinputs) / outputs
    # Normalize gradient
    dinputs3 = dinputs / samples

    """ Calculate gradients in regard to inputs, weights and biases """

    dW3 = np.dot(hidden['z2'].T, dinputs3)
    db3 = np.sum(dinputs3, axis=0, keepdims=True)
    dinputs2 = np.dot(dinputs3, W3.T)
    dinputs2 = drelu(hidden['y2'], dinputs2)
    dW2 = np.dot(hidden['z1'].T, dinputs2)
    db2 = np.sum(dinputs2, axis=0, keepdims=True)
    dinputs1 = np.dot(dinputs2, W2.T)
    dinputs1 = drelu(hidden['y1'], dinputs1)
    dW1 = np.dot(hidden['inputs'].T, dinputs1)
    db1 = np.sum(dinputs1, axis=0, keepdims=True)

    return {'W1': dW1, 'Bias_W1': db1, 'W2': dW2, 'Bias_W2': db2, 'W3': dW3, 'Bias_W3': db3}


def check_grads(model, inputs, targets, grads):
    """
        Checks if backward propagation is implemented correctly
        Arguments:
        delta -- tiny shift to the input to compute approximated gradient with formula(1)
        Returns:
        difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    param_keys, params, gradients, grad_keys = [], [], [], []
    for key, item in model.items():
        param_keys.append(key)
        params.append(item)
    gradients = []
    for k, v in grads.items():
        gradients.append(v)

    num_checks, delta = 10, 1e-5
    for param, dparam, name in zip(params, gradients,
                                   param_keys):
        s0 = dparam.shape
        s1 = param.shape
        assert s0 == s1, f'Error dims dont match: {s0} and {s1}.'
        print(name)
        for i in range(num_checks):
            ri = int(uniform(0, param.size))
            # evaluate cost at [x + delta] and [x - delta]
            old_val = param.flat[ri]
            param.flat[ri] = old_val + delta
            _, values = forward(inputs)
            cg0 = loss_fun(values, targets)
            param.flat[ri] = old_val - delta
            _, values = forward(inputs)
            cg1 = loss_fun(values, targets)
            param.flat[ri] = old_val  # reset old value for this parameter
            # fetch both numerical and analytic gradient
            grad_analytic = dparam.flat[ri]
            grad_numerical = (cg0 - cg1) / (2 * delta)
            if grad_analytic == 0.0 and grad_numerical == 0.0:
                rel_error = 0.0
            else:
                rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
            # rel_error should be on order of 1e-7 or less

            if rel_error > delta:
                print("\033[93m" + "There might be a mistake in backward propagation! difference = " + str(
                    rel_error) + "\033[0m")
            else:
                print("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(
                    rel_error) + "\033[0m")


""" Model Params """
seq_len = 10
hidden_units = 200
in_features = seq_len
out_features = 1

""" Weight Initialization """
W1 = np.random.randn(hidden_units, in_features) / np.sqrt(in_features) # Xavier Initialization
W1 = W1.T
Bias_W1 = np.array([np.zeros(hidden_units)])
W2 = np.random.randn(hidden_units, hidden_units) / np.sqrt(hidden_units) # Xavier Initialization
W2 = W2.T
Bias_W2 = np.array([np.zeros(hidden_units)])
W3 = np.random.randn(out_features, hidden_units) / np.sqrt(hidden_units) # Xavier Initialization
W3 = W3.T
Bias_W3 = np.array([np.zeros(out_features)])

model = {'W1': W1, 'Bias_W1': Bias_W1, 'W2': W2, 'Bias_W2': Bias_W2, 'W3': W3, 'Bias_W3': Bias_W3}
grad_buffer = init_grad_buffer(model)

dataset = pd.read_csv('datasets/DummyStockData.csv').to_numpy()
asks = dataset[:, 1]

length = len(asks)
last_10pct = np.array(length * 0.05, dtype=np.int)

train_data = asks[:length-last_10pct]
test_data = asks[length-last_10pct:]

scaler = preprocessing.MinMaxScaler()
test_scaler = preprocessing.MinMaxScaler()

train_data = scaler.fit_transform(np.expand_dims(train_data, axis=1))
test_data = test_scaler.fit_transform(np.expand_dims(test_data, axis=1))

X_train, y_train = create_sequences(train_data, seq_length=seq_len)
X_test, y_test = create_sequences(test_data, seq_length=seq_len)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

""" Training Params """
EPOCHS = 1000
lr = 0.002
smooth_acc = 0
smooth_loss = 0
check_gradients = False
step = 0
update_every = 1
accuracy_precision = np.std(y_train) / 10

print(f"{bcolors.OKBLUE}\nStarting Training Phase{bcolors.ENDC}")
time.sleep(2)

for epoch in range(EPOCHS):
        if check_gradients:
            hidden, values = forward(X_train)
            grads = backward(values, hidden, y_train)
            print(f'Checking gradient implementation')
            difference = check_grads(model, X_train, y_train, grads)
            check_gradients = False
            time.sleep(5)

        hidden, values = forward(X_train)
        loss = loss_fun(values, y_train)

        accuracy = np.mean(np.absolute(values - y_train) <
                                 accuracy_precision)
        if epoch == 0:
            smooth_loss = loss
            smooth_acc = accuracy
        else:
            smooth_loss = smooth_loss * 0.99 + loss * 0.01
            smooth_acc = smooth_acc * 0.99 + accuracy * 0.01

        grads = backward(values, hidden, y_train)

        for k, v in grad_buffer.items():
            grad_buffer[k] += grads[k]

        step += 1

        if not step % update_every:
            optimize(model, grad_buffer, lr=lr)
            grad_buffer = init_grad_buffer(model)

        print(f'epoch: {epoch}, ' +
              f'step: {step}, ' +
              f'smooth acc: {smooth_acc:.3f}, ' +
              f'smooth loss: {smooth_loss:.3f}')

plt.plot(values)
plt.plot(y_train)
plt.show()


print(f"{bcolors.OKBLUE}\nTesting the Model{bcolors.ENDC}")
time.sleep(2)

hidden, test_values = forward(X_test)
loss = loss_fun(test_values, y_test)
accuracy = np.mean(np.absolute(test_values - y_test) <
                   accuracy_precision)

print(f"\ntesting acc: {accuracy:.3f}, loss: {loss:.3f}")

plt.plot(test_values)
plt.plot(y_test)
plt.show()
