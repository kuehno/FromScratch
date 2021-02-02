import numpy as np
import pickle
import time
from random import uniform
import torch
from torchvision import datasets
import torchvision


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


def prepro(x):
    """ preprocessing for images only"""
    return x.astype(np.float).ravel() / 255


def softmax(x):
    exp_vals = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    return probs


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
    y3 = np.dot(z2, model['W3']) + model['Bias_W3']
    probs = softmax(y3)

    return {'inputs': inputs, 'y1': y1, 'z1': z1, 'y2': y2, 'z2': z2, 'y3': y3}, probs


def loss_fun(probs, targets):
    samples = len(probs)

    if len(targets.shape) == 1:
        target_preds = probs[range(samples), targets]

    # Mask for one-hot encoded labels
    elif len(targets.shape) == 2:
        target_preds = np.sum(probs * targets, axis=1)

    # Losses
    negative_log_probs = -np.log(target_preds)
    loss = np.mean(negative_log_probs)

    return loss


def backward(probs, hidden, targets):
    samples = len(probs)
    dinputs = probs.copy()

    # Calculate gradient
    dinputs[range(samples), targets] -= 1
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
            _, probs = forward(inputs)
            cg0 = loss_fun(probs, targets)
            param.flat[ri] = old_val - delta
            _, probs = forward(inputs)
            cg1 = loss_fun(probs, targets)
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
hidden_units = 100
in_features = 28*28
out_features = 10

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

train_loader = torch.utils.data.DataLoader(datasets.MNIST('/home/niklaskuehn/Desktop/Python and Machine Learning/FromScratch/Datasets/MNIST Digits/', train=True, transform=torchvision.transforms.ToTensor()), batch_size=64)

""" Training Params """
EPOCHS = 5
lr = 0.01
smooth_acc = 0
smooth_loss = 0
check_gradients = False
step = 0
update_every = 10

print(f"{bcolors.OKBLUE}\nStarting Training Phase{bcolors.ENDC}")
time.sleep(2)

for epoch in range(EPOCHS):
    for batch_idx, (X_train, y_train) in enumerate(train_loader):
        X_train = np.array(X_train).reshape(-1, 28*28)
        y_train = np.array(y_train)

        if check_gradients:
            hidden, probs = forward(X_train)
            grads = backward(probs, hidden, y_train)
            print(f'Checking gradient implementation')
            difference = check_grads(model, X_train, y_train, grads)
            check_gradients = False
            time.sleep(5)

        hidden, probs = forward(X_train)
        loss = loss_fun(probs, y_train)
        preds = np.argmax(probs, axis=1)
        if len(y_train.shape) == 2:
            y_train = np.argmax(y_train, axis=1)

        accuracy = np.mean(preds==y_train)
        smooth_acc = smooth_acc * 0.99 + accuracy * 0.01
        smooth_loss = smooth_loss * 0.99 + loss * 0.01

        grads = backward(probs, hidden, y_train)

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

test_loader = torch.utils.data.DataLoader(datasets.MNIST('/home/niklaskuehn/Desktop/Python and Machine Learning/FromScratch/Datasets/MNIST Digits/', train=False, transform=torchvision.transforms.ToTensor()), batch_size=10000)


print(f"{bcolors.OKBLUE}\nTesting the Model{bcolors.ENDC}")
time.sleep(2)

for batch_idx, (X_test, y_test) in enumerate(test_loader):
    X_test = np.array(X_test).reshape(-1, 28*28)
    y_test = np.array(y_test)

    hidden, probs = forward(X_test)
    loss = loss_fun(probs, y_test)
    preds = np.argmax(probs, axis=1)
    if len(y_test.shape) == 2:
        y_train = np.argmax(y_test, axis=1)

    accuracy = np.mean(preds==y_test)
    print(f"\ntesting acc: {accuracy:.3f}, loss: {loss:.3f}")
