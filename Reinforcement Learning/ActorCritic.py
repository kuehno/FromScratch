import numpy as np
import time
from random import uniform
import cv2


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


class Player:
    def __init__(self, pos):
        """ Starting position in the center of the screen """
        self.x = pos[0]
        self.y = pos[1]

    def __str__(self):
        return f"[{self.x}, {self.y}]"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def move(self, action, SIZE):
        if action == 0:
            self.x += 1
        if action == 1:
            self.x -= 1
        if action == 2:
            self.y += 1
        if action == 3:
            self.y -= 1

        if self.x < 0:
            self.x = 0
        elif self.x > SIZE - 1:
            self.x = SIZE - 1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE - 1:
            self.y = SIZE - 1


class GameEnvironment:
    SIZE = 10
    PLAYER_N = 175
    PLAYER_N_1D = 50
    PLAYER_POS = (0, 0)
    GOAL_N = 100
    GOAL_POS = ((SIZE - 1) // 2 , (SIZE - 1) // 2)
    NUM_ENEMIES = 5 # change to make the environment contain enemies
    positions = {}
    for i in range(NUM_ENEMIES):
        positions[f'{i}'] = (np.random.randint(1, SIZE-1), np.random.randint(1, SIZE-1))
    ENEMY1_POS = (1, 2)
    ENEMY2_POS = (2, 2)
    ENEMY_N = 255


    def reset(self):
        self.player = Player(self.PLAYER_POS)
        self.enemies = {}
        for i in range(self.NUM_ENEMIES):
            self.enemies[f'Enemy{i}'] = Player(self.positions[f'{i}'])
        self.episode_step = 0

        obs = self.get_observation()

        return obs

    def step(self, action):
        self.player.move(action, self.SIZE)
        self.episode_step += 1

        obs = self.get_observation()

        env = np.zeros((self.SIZE, self.SIZE),dtype=np.float)

        for i in range(self.NUM_ENEMIES):
            env[(self.enemies[f"Enemy{i}"].x, self.enemies[f"Enemy{i}"].y)] = self.ENEMY_N

        if env[self.player.x, self.player.y] == 255:
            reward = -1.0
            done = True
        elif (self.player.x, self.player.y) == self.GOAL_POS:
            reward = 1.0
            done = True
        else:
            reward = -0.02
            done = False

        if self.episode_step >= 100:
            reward = -1.0
            done = True

        return obs, reward, done

    def get_observation(self):
        env = np.zeros((self.SIZE, self.SIZE), dtype=np.float)

        env[self.GOAL_POS] = self.GOAL_N
        for i in range(self.NUM_ENEMIES):
            env[(self.enemies[f"Enemy{i}"].x, self.enemies[f"Enemy{i}"].y)] = self.ENEMY_N
        env[self.player.x][self.player.y] = self.PLAYER_N_1D

        return env

    def render(self):
        env = np.zeros((self.SIZE, self.SIZE, 3),dtype=np.float)

        env[self.GOAL_POS] = [0, self.GOAL_N, 0]
        for i in range(self.NUM_ENEMIES):
            env[(self.enemies[f"Enemy{i}"].x, self.enemies[f"Enemy{i}"].y)] = [0, 0, self.ENEMY_N]
        env[self.player.x][self.player.y] = [self.PLAYER_N, 0, 0]

        img = cv2.resize(env, (400, 400), interpolation=cv2.INTER_AREA)
        cv2.imshow("Own Network Env", img)
        cv2.waitKey(20)


def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


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


class Adam:
    def __init__(self, model, lr, beta_1=0.9, beta_2=0.999):
        self.momentums = {k: np.zeros_like(v) for k, v in model.items()}
        self.momentums_corrected = {k: np.zeros_like(v) for k, v in model.items()}
        self.cache = {k: np.zeros_like(v) for k, v in model.items()}
        self.cache_corrected = {k: np.zeros_like(v) for k, v in model.items()}
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iterations = 0

    def update_params(self, model, grad_buffer):
        for k, v in model.items():
            g = grad_buffer[k]  # gradient
            self.momentums[k] = self.beta_1 * self.momentums[k] + (1 - self.beta_1) * g
            self.momentums_corrected[k] = self.momentums[k] / (1 - self.beta_1 ** (self.iterations + 1))
            # Update cache with squared current gradients
            self.cache[k] = self.beta_2 * self.cache[k] + (1 - self.beta_2) * g ** 2
            # Get corrected cache
            self.cache_corrected[k] = self.cache[k] / (1 - self.beta_2 ** (self.iterations + 1))
            # Vanilla SGD parameter update + normalization
            # with square rooted cache
            model[k] += -self.lr * self.momentums_corrected[k] / (
                    np.sqrt(self.cache_corrected[k]) + 1e-7)
        self.iterations += 1


def init_grad_buffer(model):
    grad_buffer = {}
    for k, v in model.items():
        grad_buffer[f'{k}'] = np.zeros_like(v)
    return grad_buffer


def reset_memory(num_layers):
    memory = {**{'inputs': []}, **{f'y{i}': [] for i in range(1, num_layers)},
            **{f'z{i}': [] for i in range(1, num_layers)}}
    memory_step = 0
    return memory, memory_step


def stack_memory(memory, new_values, memory_step):
    if memory['inputs'] == []:
        for k in new_values:
            memory[k] = new_values[k]
    else:
        for k in new_values:
            if k == 'inputs':
                if memory_step == 1:
                    memory[k] = np.stack((memory[k], new_values[k]), axis=0)
                else:
                    inputs = new_values[k].reshape(1, -1)
                    memory[k] = np.concatenate((memory[k], inputs), axis=0)
            else:
                memory[k] = np.concatenate((memory[k], new_values[k].reshape(1, -1)), axis=0)
    memory_step += 1
    return memory, memory_step


def forward(inputs, model):
    y1 = np.dot(inputs, model['W1']) + model['Bias_W1']
    z1 = relu(y1)
    y2 = np.dot(z1, model['W2']) + model['Bias_W2']
    z2 = relu(y2)
    y3 = np.dot(z2, model['W3']) + model['Bias_W3']
    z3 = relu(y3)
    y4 = np.dot(z3, model['W4']) + model['Bias_W4']
    preds = y4

    return {'inputs': inputs, 'y1': y1, 'y2': y2, 'y3': y3, 'z1': z1, 'z2': z2, 'z3': z3}, preds


def a_loss_fun(probs, targets, rewards):
    samples = len(probs)

    if len(targets.shape) == 1:
        target_preds = probs[range(samples), targets]

    # Mask for one-hot encoded labels
    elif len(targets.shape) == 2:
        target_preds = np.sum(probs * targets, axis=1)

    # Losses
    negative_log_probs = -np.log(target_preds) * np.hstack(rewards)
    loss = np.mean(negative_log_probs)

    return loss


def c_loss_fun(values, rewards):
    loss = np.mean((rewards - values) ** 2) #MSE Loss
    return loss


def backward(dinputs, hidden, model):
    """ Calculate gradients in regard to inputs, weights and biases """

    dW4 = np.dot(hidden['z3'].T, dinputs)
    db4 = np.sum(dinputs, axis=0, keepdims=True)
    dinputs3 = np.dot(dinputs, model['W4'].T)
    dinputs3 = drelu(hidden['y3'], dinputs3)
    dW3 = np.dot(hidden['z2'].T, dinputs3)
    db3 = np.sum(dinputs3, axis=0, keepdims=True)
    dinputs2 = np.dot(dinputs3, model['W3'].T)
    dinputs2 = drelu(hidden['y2'], dinputs2)
    dW2 = np.dot(hidden['z1'].T, dinputs2)
    db2 = np.sum(dinputs2, axis=0, keepdims=True)
    dinputs1 = np.dot(dinputs2, model['W2'].T)
    dinputs1 = drelu(hidden['y1'], dinputs1)
    dW1 = np.dot(hidden['inputs'].T, dinputs1)
    db1 = np.sum(dinputs1, axis=0, keepdims=True)

    return {'W1': dW1, 'Bias_W1': db1, 'W2': dW2, 'Bias_W2': db2, 'W3': dW3, 'Bias_W3': db3, 'W4': dW4, 'Bias_W4': db4}


def check_grads_actor(model, inputs, targets, rewards, grads):
    """
        Checks if backward propagation is implemented correctly
        Arguments:
        delta -- tiny shift to the input to compute approximated gradient with formula(1)
        Returns:
        difference -- difference (2) between the approximated gradient and the backward propagation gradient

        Inspired by Andrej Karpathy - min-char-rnn -> https://gist.github.com/karpathy/d4dee566867f8291f086
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
            old_val = param.flat[ri]
            param.flat[ri] = old_val + delta
            _, probs = forward(inputs, model=model)
            probs = softmax(probs)
            loss_plus = a_loss_fun(probs, targets, rewards)
            param.flat[ri] = old_val - delta
            _, probs = forward(inputs, model=model)
            probs = softmax(probs)
            loss_minus = a_loss_fun(probs, targets, rewards)
            param.flat[ri] = old_val  # reset old value
            grad_analytic = dparam.flat[ri]
            grad_numerical = (loss_plus - loss_minus) / (2 * delta)
            if grad_analytic == 0.0 and grad_numerical == 0.0:
                rel_error = 0.0
            else:
                rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)

            if rel_error > delta:
                print(f"{bcolors.FAIL}There might be a mistake in backward propagation! difference = {rel_error}{bcolors.ENDC}")
            else:
                print(f"{bcolors.OKGREEN}Your backward propagation works perfectly fine! difference = {rel_error}{bcolors.ENDC}")


def check_grads_critic(model, inputs, rewards, grads):
    """
        Checks if backward propagation is implemented correctly
        Arguments:
        delta -- tiny shift to the input to compute approximated gradient with formula(1)
        Returns:
        difference -- difference (2) between the approximated gradient and the backward propagation gradient

        Inspired by Andrej Karpathy - min-char-rnn -> https://gist.github.com/karpathy/d4dee566867f8291f086
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
            old_val = param.flat[ri]
            param.flat[ri] = old_val + delta
            _, values = forward(inputs, model=model)
            loss_plus = c_loss_fun(values, rewards)
            param.flat[ri] = old_val - delta
            _, values = forward(inputs, model=model)
            loss_minus = c_loss_fun(values, rewards)
            param.flat[ri] = old_val  # reset old value
            grad_analytic = dparam.flat[ri]
            grad_numerical = (loss_plus - loss_minus) / (2 * delta)
            if grad_analytic == 0.0 and grad_numerical == 0.0:
                rel_error = 0.0
            else:
                rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)

            if rel_error > delta:
                print(f"{bcolors.FAIL}There might be a mistake in backward propagation! difference = {rel_error}{bcolors.ENDC}")
            else:
                print(f"{bcolors.OKGREEN}Your backward propagation works perfectly fine! difference = {rel_error}{bcolors.ENDC}")


env = GameEnvironment()
print(f"{bcolors.OKGREEN}\nInitial game state{bcolors.ENDC}")
env.reset()
env.render()
time.sleep(2)


def init_model(hidden_dim, in_dim, out_dim):
    """ Weight Initialization """
    W1 = np.random.randn(hidden_dim, in_dim) / np.sqrt(in_dim) # Xavier Initialization
    W1 = W1.T
    Bias_W1 = np.array([np.zeros(hidden_dim)])
    W2 = np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim) # Xavier Initialization
    W2 = W2.T
    Bias_W2 = np.array([np.zeros(hidden_dim)])
    W3 = np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim) # Xavier Initialization
    W3 = W3.T
    Bias_W3 = np.array([np.zeros(hidden_dim)])
    W4 = np.random.randn(out_dim, hidden_dim) / np.sqrt(hidden_dim) # Xavier Initialization
    W4 = W4.T
    Bias_W4 = np.array([np.zeros(out_dim)])

    return {'W1': W1, 'Bias_W1': Bias_W1, 'W2': W2, 'Bias_W2': Bias_W2, 'W3': W3, 'Bias_W3': Bias_W3, 'W4': W4, 'Bias_W4': Bias_W4}


""" Model Params """
# Actor
hidden_units = 64
in_features = env.SIZE * env.SIZE
a_out_features = 4
# Critic
c_out_features = 1


""" Training Params """
EPOCHS = 10000
lr = 0.0005
smooth_rewards = 0
smooth_loss = 0
step = 0
update_every = 10
check_gradients = False
render = True


actor = init_model(hidden_dim=hidden_units, in_dim=in_features, out_dim=a_out_features)
critic = init_model(hidden_dim=hidden_units, in_dim=in_features, out_dim=c_out_features)

a_grad_buffer = init_grad_buffer(actor)
c_grad_buffer = init_grad_buffer(critic)

a_optimizer = Adam(actor, lr=lr)
c_optimizer = Adam(critic, lr=lr)

print(f"{bcolors.OKBLUE}\nStarting Training Phase{bcolors.ENDC}")
time.sleep(2)

for epoch in range(EPOCHS):
    a_memory, a_memory_step = reset_memory(num_layers=4)
    c_memory, c_memory_step = reset_memory(num_layers=4)
    targets, rewards, ep_probs, ep_values = [], [], [], []
    ep_rewards = 0
    done = False
    obs = prepro(env.reset())
    while not done:
        a_hidden, preds = forward(obs, model=actor)
        c_hidden, values = forward(obs, model=critic)
        probs = softmax(preds)
        a_memory, a_memory_step = stack_memory(a_memory, a_hidden, memory_step=a_memory_step)
        c_memory, c_memory_step = stack_memory(c_memory, c_hidden, memory_step=c_memory_step)
        action = np.random.choice(4, p=probs.ravel())
        ep_probs.append(probs)
        ep_values.append(values)

        obs, reward, done = env.step(action)
        obs = prepro(obs)
        ep_rewards += reward

        targets.append(action)
        rewards.append(reward)

        if smooth_rewards >= 0.9 and render or epoch > 9000 and render:
            env.render()

    targets = np.hstack(targets)
    rewards = np.vstack(rewards)
    ep_probs = np.vstack(ep_probs)
    ep_values = np.vstack(ep_values)

    rewards = discount_rewards(rewards, gamma=0.98)
    # reward normalization
    # rewards -= np.mean(rewards)
    # rewards /= np.std(rewards)

    if check_gradients:
        """ first check gradients for actor """
        a_hidden, preds = forward(a_memory['inputs'], model=actor)
        probs = softmax(preds)
        c_hidden, values = forward(c_memory['inputs'], model=critic)

        advantages = rewards - values

        """ calculate dloss for actor """
        samples = len(probs)
        dinputs = probs.copy()

        # Calculate gradient
        dinputs[range(samples), targets] -= 1
        dinputs *= advantages
        # Normalize gradient
        dinputs = dinputs / samples

        grads = backward(dinputs, a_hidden, model=actor)
        print(f'{bcolors.OKBLUE}\nChecking actor gradient implementation{bcolors.ENDC}')
        check_grads_actor(actor, a_memory['inputs'], targets, advantages, grads)

        """ checking gradients for critic """

        """ calculate dloss for critic """
        outputs = len(values[0])

        # Calculate gradient
        dinputs = -2 * (rewards - values) / outputs
        # Normalize gradient
        dinputs = dinputs / samples

        grads = backward(dinputs, c_hidden, model=critic)
        print(f'{bcolors.OKBLUE}\nChecking critic gradient implementation{bcolors.ENDC}')
        check_grads_critic(critic, c_memory['inputs'], rewards, grads)

        check_gradients = False
        time.sleep(5)

    a_loss = a_loss_fun(ep_probs, targets, rewards)
    c_loss = c_loss_fun(ep_values, rewards)
    smooth_rewards = smooth_rewards * 0.99 + ep_rewards * 0.01
    smooth_loss = smooth_loss * 0.99 + (a_loss * 0.01 + c_loss * 0.01)

    """ actor grads """
    samples = len(ep_probs)
    dinputs = ep_probs.copy()

    advantages = rewards - ep_values

    # Calculate gradient
    dinputs[range(samples), targets] -= 1
    dinputs *= advantages
    # Normalize gradient
    dinputs = dinputs / samples
    a_grads = backward(dinputs, a_memory, model=actor)

    for k, v in a_grad_buffer.items():
        a_grad_buffer[k] += a_grads[k]

    """ critic grads """
    outputs = len(ep_values[0])

    # Calculate gradient
    dinputs = -2 * (rewards - ep_values) / outputs
    # Normalize gradient
    dinputs = dinputs / samples

    c_grads = backward(dinputs, c_memory, model=critic)

    for k, v in c_grad_buffer.items():
        c_grad_buffer[k] += c_grads[k]

    step += 1

    if not step % update_every:
        a_optimizer.update_params(actor, a_grad_buffer)
        c_optimizer.update_params(critic, c_grad_buffer)
        a_grad_buffer = init_grad_buffer(actor)
        c_grad_buffer = init_grad_buffer(critic)

    print(f'epoch: {epoch}, ' +
          f'smooth rewards: {smooth_rewards:.3f}, ' +
          f'smooth loss: {smooth_loss:.3f}')
