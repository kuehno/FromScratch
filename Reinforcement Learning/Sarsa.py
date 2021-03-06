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
    SIZE = 6
    PLAYER_N = 175
    PLAYER_N_1D = 50
    PLAYER_POS = (0, 0)
    GOAL_N = 100
    GOAL_POS = (SIZE - 1, SIZE - 1)
    NUM_ENEMIES = 2 # change to make the environment contain enemies
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
            reward = 0.0
            done = False

        if self.episode_step >= 200:
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


def relu(x):
    return np.maximum(0, x)


def drelu(inputs, dinputs):
    dinputs[inputs <= 0] = 0
    return dinputs


def optimize(model, grad_buffer, lr):
    """ Vanilla Stochastic Gradient Descent """
    for k, v in model.items():
        model[k] += -lr * grad_buffer[k]


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


def forward(inputs):
    y1 = np.dot(inputs, model['W1']) + model['Bias_W1']
    z1 = relu(y1)
    y2 = np.dot(z1, model['W2']) + model['Bias_W2']
    z2 = relu(y2)
    y3 = np.dot(z2, model['W3']) + model['Bias_W3']
    z3 = relu(y3)
    y4 = np.dot(z3, model['W4']) + model['Bias_W4']
    values = y4

    return {'inputs': inputs, 'y1': y1, 'y2': y2, 'y3': y3, 'z1': z1, 'z2': z2, 'z3': z3}, values


def loss_fun(actual_qs, target_qs):
    loss = np.mean((target_qs - actual_qs) ** 2)  # new target qs - target qs ** 2
    return loss


def backward(actual_qs, hidden, target_qs):
    samples = len(actual_qs)
    outputs = len(actual_qs[0])

    # Calculate gradient
    dinputs = -2 * (target_qs - actual_qs) / outputs
    # Normalize gradient
    dinputs4 = dinputs / samples

    """ Calculate gradients in regard to inputs, weights and biases """

    dW4 = np.dot(hidden['z3'].T, dinputs4)
    db4 = np.sum(dinputs4, axis=0, keepdims=True)
    dinputs3 = np.dot(dinputs4, W4.T)
    dinputs3 = drelu(hidden['y3'], dinputs3)
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

    return {'W1': dW1, 'Bias_W1': db1, 'W2': dW2, 'Bias_W2': db2, 'W3': dW3, 'Bias_W3': db3, 'W4': dW4, 'Bias_W4': db4}


def check_grads(model, inputs, targets, new_ep_values, new_targets, rewards, grads):
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
            _, values_plus = forward(inputs)

            samples = len(values_plus)
            test_target_qs = new_ep_values.copy()
            rewards = np.hstack(rewards)
            test_act_target_qs = test_target_qs[range(samples), new_targets]
            dones = np.zeros_like(test_act_target_qs)
            dones[-1] = 1
            test_act_target_qs = rewards + 0.99 * (1 - dones) * test_act_target_qs
            test_target_qs = values_plus.copy()
            test_target_qs[range(samples), targets] = test_act_target_qs

            loss_plus = loss_fun(values_plus, test_target_qs)
            param.flat[ri] = old_val - delta
            _, values_minus = forward(inputs)

            samples = len(values_minus)
            test_target_qs = new_ep_values.copy()
            rewards = np.hstack(rewards)
            test_act_target_qs = test_target_qs[range(samples), new_targets]
            dones = np.zeros_like(test_act_target_qs)
            dones[-1] = 1
            test_act_target_qs = rewards + 0.99 * (1 - dones) * test_act_target_qs
            test_target_qs = values_minus.copy()
            test_target_qs[range(samples), targets] = test_act_target_qs

            loss_minus = loss_fun(values_minus, test_target_qs)
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
W3 = np.random.randn(hidden_units, hidden_units) / np.sqrt(hidden_units) # Xavier Initialization
W3 = W3.T
Bias_W3 = np.array([np.zeros(hidden_units)])
W4 = np.random.randn(out_features, hidden_units) / np.sqrt(hidden_units) # Xavier Initialization
W4 = W4.T
Bias_W4 = np.array([np.zeros(out_features)])

model = {'W1': W1, 'Bias_W1': Bias_W1, 'W2': W2, 'Bias_W2': Bias_W2, 'W3': W3, 'Bias_W3': Bias_W3, 'W4': W4, 'Bias_W4': Bias_W4}
grad_buffer = init_grad_buffer(model)

""" Training Params """
EPOCHS = 50000
lr = 0.0002
epsilon = 0.9
epsilon_decay = 0.999
smooth_rewards = 0
smooth_loss = 0
step = 0
update_every = 1
check_gradients = False
render = True

optimizer = Adam(model, lr=lr)

print(f"{bcolors.OKBLUE}\nStarting Training Phase{bcolors.ENDC}")
time.sleep(2)

for epoch in range(EPOCHS):
    memory, memory_step = reset_memory(num_layers=4)
    targets, new_targets, rewards, ep_values, new_ep_values = [], [], [], [], []
    ep_rewards = 0
    done = False
    obs = prepro(env.reset())
    while not done:
        hidden, values = forward(obs)
        ep_values.append(values)
        memory, memory_step = stack_memory(memory, hidden, memory_step=memory_step)
        if np.random.rand() > epsilon:
            action = np.argmax(values)
        else:
            action = np.random.randint(0, 4)

        new_obs, reward, done = env.step(action)
        new_obs = prepro(new_obs)
        ep_rewards += reward

        _, new_values = forward(new_obs)
        new_ep_values.append(new_values)
        if np.random.rand() > epsilon:
            new_action = np.argmax(new_values)
        else:
            new_action = np.random.randint(0, 4)

        targets.append(action)
        new_targets.append(new_action)
        rewards.append(reward)

        if smooth_rewards > 0.99 and render or smooth_rewards <= -0.99 and render and epoch >= 2000:
            env.render()

        obs = new_obs

    epsilon *= epsilon_decay

    targets = np.hstack(targets)
    new_targets = np.hstack(new_targets)
    rewards = np.vstack(rewards)
    ep_values = np.vstack(ep_values)
    new_ep_values = np.vstack(new_ep_values)

    # rewards = discount_rewards(rewards, gamma=0.99)
    # # reward normalization
    # rewards -= np.mean(rewards)
    # rewards /= np.std(rewards)

    """ calculate actual qs and target qs """
    samples = len(ep_values)

    actual_qs = ep_values.copy()
    target_qs = new_ep_values.copy()

    rewards = np.hstack(rewards)

    act_target_qs = target_qs[range(samples), new_targets]

    dones = np.zeros_like(act_target_qs)
    dones[-1] = 1
    act_target_qs = rewards + 0.99 * (1 - dones) * act_target_qs

    target_qs = ep_values.copy()
    target_qs[range(samples), targets] = act_target_qs

    if check_gradients:
        # hidden, values = forward(memory['inputs'])
        grads = backward(actual_qs, memory, target_qs)
        print(f'Checking gradient implementation')
        difference = check_grads(model, memory['inputs'], targets, new_ep_values, new_targets, rewards, grads)
        check_gradients = False
        time.sleep(5)

    loss = loss_fun(actual_qs, target_qs)
    if epoch == 0:
        smooth_rewards = ep_rewards
        smooth_loss = loss
    else:
        smooth_rewards = smooth_rewards * 0.99 + ep_rewards * 0.01
        smooth_loss = smooth_loss * 0.99 + loss * 0.01

    grads = backward(actual_qs, memory, target_qs)

    for k, v in grad_buffer.items():
        grads[k] = np.clip(grads[k], -1, 1)
        grad_buffer[k] += grads[k]

    step += 1

    if not step % update_every:
        optimizer.update_params(model, grad_buffer)
        grad_buffer = init_grad_buffer(model)

    print(f'epoch: {epoch}, ' +
          f'epsilon: {epsilon}, ' +
          f'smooth rewards: {smooth_rewards:.3f}, ' +
          f'smooth loss: {smooth_loss:.3f}')
