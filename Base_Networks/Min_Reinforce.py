import numpy as np
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
    SIZE = 6
    PLAYER_N = 175
    PLAYER_N_1D = 50
    PLAYER_POS = (0, 0)
    GOAL_N = 100
    GOAL_N_1D = 150
    GOAL_POS = (SIZE - 1, SIZE - 1)
    #ENEMY1_POS = (np.random.randint(1, 4), np.random.randint(1, 4))
    #ENEMY2_POS = (np.random.randint(1, 4), np.random.randint(1, 4))
    NUM_ENEMIES = 0
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
        #self.ENEMY1_POS = (np.random.randint(1, 4), np.random.randint(1, 4))
        #self.ENEMY2_POS = (np.random.randint(1, 4), np.random.randint(1, 4))
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
            reward = 0.0
            done = False

        if self.episode_step >= 50:
            reward = -1.0
            done = True

        return obs, reward, done

    def get_observation(self):
        env = np.zeros((self.SIZE, self.SIZE), dtype=np.float)

        env[self.GOAL_POS] = self.GOAL_N_1D
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
        cv2.waitKey(10)


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


def forward(inputs):
    y1 = np.dot(inputs, model['W1']) + model['Bias_W1']
    z1 = relu(y1)
    y2 = np.dot(z1, model['W2']) + model['Bias_W2']
    z2 = relu(y2)
    y3 = np.dot(z2, model['W3']) + model['Bias_W3']
    probs = softmax(y3)

    return {'inputs': inputs, 'y1': y1, 'y2': y2, 'z1': z1, 'z2': z2}, probs


def loss_fun(probs, targets, rewards):
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


def backward(probs, hidden, targets, rewards):
    samples = len(probs)
    dinputs = probs.copy()

    # Calculate gradient
    dinputs[range(samples), targets] -= 1
    dinputs *= rewards
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


def check_grads(model, inputs, targets, rewards, grads):
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
            cg0 = loss_fun(probs, targets, rewards)
            param.flat[ri] = old_val - delta
            _, probs = forward(inputs)
            cg1 = loss_fun(probs, targets, rewards)
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


env = GameEnvironment()

""" Model Params """
hidden_units = 64
in_features = env.SIZE * env.SIZE
out_features = 4

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

""" Training Params """
EPOCHS = 5000
lr = 0.02
smooth_rewards = 0
smooth_loss = 0
step = 0
update_every = 1
check_gradients = False
render = False

print(f"{bcolors.OKBLUE}\nStarting Training Phase{bcolors.ENDC}")
time.sleep(2)

for epoch in range(EPOCHS):
    memory, memory_step = reset_memory(num_layers=3)
    targets, rewards, ep_probs = [], [], []
    ep_rewards = 0
    done = False
    obs = prepro(env.reset())
    while not done:
        hidden, probs = forward(obs)
        memory, memory_step = stack_memory(memory, hidden, memory_step=memory_step)
        action = np.random.choice(4, p=probs.ravel())
        ep_probs.append(probs)

        obs, reward, done = env.step(action)
        obs = prepro(obs)
        ep_rewards += reward

        targets.append(action)
        rewards.append(reward)

        if smooth_rewards >= 0.99 and render:
            env.render()

    targets = np.hstack(targets)
    rewards = np.vstack(rewards)
    ep_probs = np.vstack(ep_probs)

    rewards = discount_rewards(rewards, gamma=0.9)

    if check_gradients:
        hidden, probs = forward(memory['inputs'])
        grads = backward(ep_probs, memory, targets, rewards)
        print(f'Checking gradient implementation')
        difference = check_grads(model, memory['inputs'], targets, rewards, grads)
        check_gradients = False
        time.sleep(5)

    loss = loss_fun(ep_probs, targets, rewards)
    smooth_rewards = smooth_rewards * 0.99 + ep_rewards * 0.01
    smooth_loss = smooth_loss * 0.99 + loss * 0.01

    grads = backward(ep_probs, memory, targets, rewards)

    for k, v in grad_buffer.items():
        grad_buffer[k] += grads[k]

    step += 1

    if not step % update_every:
        optimize(model, grad_buffer, lr=lr)
        grad_buffer = init_grad_buffer(model)

    print(f'epoch: {epoch}, ' +
          f'smooth rewards: {smooth_rewards:.3f}, ' +
          f'smooth loss: {smooth_loss:.3f}')