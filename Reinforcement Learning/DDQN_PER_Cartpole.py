import numpy as np
import time
from random import uniform
import gym


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


class PrioritizedReplayMemory:
    def __init__(self, batch_size, max_size, alpha=0.8, epsilon=0.01):
        self.size = 0
        self.batch_size = batch_size
        self.max_size = max_size
        self.keys = ['states', 'actions', 'rewards', 'new_states', 'dones', 'priorities']
        self.head = -1
        self.reset()
        self.alpha = np.full((1, ), alpha)
        self.epsilon = np.full((1, ), epsilon)

    def reset(self):
        for k in self.keys:
            setattr(self, k, [None] * self.max_size)
        self.size = 0
        self.head = -1
        self.tree = SumTree(self.max_size)

    def add_experience(self, states, actions, rewards, new_states, dones, error=10_000):
        for i in range(len(states)):
            self.head = (self.head + 1) % self.max_size
            self.states[self.head] = states[i].astype(np.float16)
            self.actions[self.head] = actions[i]
            self.rewards[self.head] = rewards[i]
            self.new_states[self.head] = new_states[i].astype(np.float16)
            self.dones[self.head] = dones[i]
            priority = self.get_priority(error)
            self.priorities[self.head] = priority
            self.tree.add(priority, self.head)

            if self.size < self.max_size:
                self.size += 1

    def get_priority(self, error):
        return np.power(error + self.epsilon, self.alpha)

    def update_priorities(self, errors):
        priorities = self.get_priority(errors)
        assert len(priorities) == self.batch_index.size
        for index, p in zip(self.batch_index, priorities):
            self.priorities[index] = p
        for p, i in zip(priorities, self.tree_index):
            self.tree.update(i, p)

    def sample_index(self, batch_size):
        batch_index = np.zeros(batch_size)
        tree_index = np.zeros(batch_size, dtype=np.int)

        for i in range(batch_size):
            s = uniform(0, self.tree.total())
            (tree_idx, p, idx) = self.tree.get(s)
            batch_index[i] = idx
            tree_index[i] = tree_idx

        batch_index = np.asarray(batch_index).astype(int)
        self.tree_index = tree_index
        return batch_index

    def sample(self):
        self.batch_index = self.sample_index(self.batch_size)
        # print(f"batch index: {self.batch_index}")
        batch = {}
        for k in self.keys:
            if k == 'states' or k == 'new_states':
                batch[k] = np.vstack(np.array(getattr(self, k))[self.batch_index])
            elif k == 'dones' or k == 'actions':
                batch[k] = np.array(getattr(self, k))[self.batch_index].astype(np.int)
            else:
                batch[k] = np.array(getattr(self, k))[self.batch_index]
        return batch


class SumTree:
    '''
    Helper class for PrioritizedReplay

    This implementation is, with minor adaptations, Jaromír Janisch's. The license is reproduced below.
    For more information see his excellent blog series "Let's make a DQN" https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/

    MIT License

    Copyright (c) 2018 Jaromír Janisch
    '''
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Stores the priorities and sums of priorities
        self.indices = np.zeros(capacity)  # Stores the indices of the experiences

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, index):
        idx = self.write + self.capacity - 1

        self.indices[self.write] = index
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        assert s <= self.total()
        idx = self._retrieve(0, s)
        indexIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.indices[indexIdx])


def forward(inputs, net):
    y1 = np.dot(inputs, net['W1']) + net['Bias_W1']
    z1 = relu(y1)
    y2 = np.dot(z1, net['W2']) + net['Bias_W2']
    z2 = relu(y2)
    y3 = np.dot(z2, net['W3']) + net['Bias_W3']
    z3 = relu(y3)
    y4 = np.dot(z3, net['W4']) + net['Bias_W4']
    values = y4

    return {'inputs': inputs, 'y1': y1, 'y2': y2, 'y3': y3, 'z1': z1, 'z2': z2, 'z3': z3}, values


def loss_fun(actual_qs, target_qs):
    loss = np.mean((target_qs - actual_qs) ** 2)  # new target qs - target qs ** 2
    return loss


def backward(actual_qs, hidden, target_qs, model):
    samples = len(actual_qs)
    outputs = len(actual_qs[0])

    # Calculate gradient
    dinputs = -2 * (target_qs - actual_qs) / outputs
    # Normalize gradient
    dinputs4 = dinputs / samples

    """ Calculate gradients in regard to inputs, weights and biases """

    dW4 = np.dot(hidden['z3'].T, dinputs4)
    db4 = np.sum(dinputs4, axis=0, keepdims=True)
    dinputs3 = np.dot(dinputs4, model['W4'].T)
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


def check_grads(model, batch, q_target, grads):
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

            _, actual_qs_plus = forward(batch['states'].copy(), model)
            loss_plus = loss_fun(actual_qs_plus, q_target)
            param.flat[ri] = old_val - delta

            _, actual_qs_minus = forward(batch['states'].copy(), model)
            loss_minus = loss_fun(actual_qs_minus, q_target)
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


env = gym.make('CartPole-v0')
print(f"{bcolors.OKGREEN}\nInitial game state{bcolors.ENDC}")
env.reset()
env.render()
time.sleep(2)

""" Model Params """
hidden_units = 100
in_features = 4
out_features = env.action_space.n

base_model = init_model(hidden_dim=hidden_units, in_dim=in_features, out_dim=out_features)
target_model = base_model.copy()
eval_net = base_model.copy()

grad_buffer = init_grad_buffer(base_model)

""" Training Params """
EPOCHS = 50000
batch_size = 200
min_memory_size = 5_000
lr = 0.001
gamma = 0.99 # discount factor
epsilon = 0.99 # randomness of actions
min_epsilon = 0.02
epsilon_decay = 0.995
smooth_rewards = 0
smooth_loss = 0
step = 0
update_weights_every = 1
update_target_net = 100
check_gradients = False
render = True
DoubleDQN = True

optimizer = Adam(base_model, lr=lr)
replay_memory = PrioritizedReplayMemory(batch_size=batch_size, max_size=200_000)

print(f"{bcolors.OKBLUE}\nStarting Training Phase{bcolors.ENDC}")
time.sleep(2)

for epoch in range(EPOCHS):
    observations, actions, rewards, new_observations, dones = [], [], [], [], []
    ep_rewards = 0
    done = False
    obs = env.reset()
    while not done:
        hidden, values = forward(obs, base_model)
        if np.random.rand() > epsilon:
            action = np.argmax(values)
        else:
            action = np.random.randint(0, env.action_space.n)

        new_obs, reward, done, _ = env.step(action)
        new_obs = new_obs
        ep_rewards += reward

        _, new_values = forward(new_obs, base_model)

        if smooth_rewards > 195 and render:
            # print(f"values: {values}")
            env.render()

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        new_observations.append(new_obs)
        dones.append(done)

        obs = new_obs

    observations = np.vstack(observations)
    actions = np.hstack(actions)
    rewards = np.hstack(rewards)
    new_observations = np.vstack(new_observations)
    dones = np.hstack(dones)

    replay_memory.add_experience(observations, actions, rewards, new_observations, dones)

    if epsilon > min_epsilon:
        epsilon *= epsilon_decay
    else:
        epsilon = min_epsilon

    if replay_memory.size >= min_memory_size:
        if check_gradients:
            batch = replay_memory.sample()

            hidden, actual_qs = forward(batch['states'], base_model)
            _, target_qs = forward(batch['new_states'], target_model)
            _, eval_qs = forward(batch['new_states'], eval_net)

            samples = len(actual_qs)

            rewards = batch['rewards']

            online_actions = np.argmax(target_qs, axis=1)
            act_target_qs = eval_qs[range(samples), online_actions]
            dones = batch['dones']
            act_target_qs = rewards + gamma * (1 - dones) * act_target_qs
            target_qs = actual_qs.copy()
            target_qs[range(samples), batch['actions']] = act_target_qs

            grads = backward(actual_qs, hidden, target_qs, base_model)
            print(f'Checking gradient implementation')
            difference = check_grads(base_model, batch, target_qs, grads)
            check_gradients = False
            time.sleep(5)

        batch = replay_memory.sample()
        if DoubleDQN:
            """If Double DQN use current model to choose actions and eval_net to calculate (lagged) Q-Value-Estimates"""
            target_model = base_model.copy()

        hidden, actual_qs = forward(batch['states'], base_model)
        _, target_qs = forward(batch['new_states'], target_model)
        _, eval_qs = forward(batch['new_states'], eval_net)

        """ calculate actual qs and target qs """
        samples = len(actual_qs)

        rewards = batch['rewards']

        online_actions = np.argmax(target_qs, axis=1)
        act_target_qs = eval_qs[range(samples), online_actions]
        dones = batch['dones']
        act_target_qs = rewards + gamma * (1 - dones) * act_target_qs
        target_qs = actual_qs.copy()
        target_qs[range(samples), batch['actions']] = act_target_qs

        errors = np.abs(act_target_qs - actual_qs[range(samples), batch['actions']])
        replay_memory.update_priorities(errors)

        loss = loss_fun(actual_qs, target_qs)
        if epoch == 0:
            smooth_rewards = ep_rewards
            smooth_loss = loss
        else:
            smooth_rewards = smooth_rewards * 0.99 + ep_rewards * 0.01
            smooth_loss = smooth_loss * 0.99 + loss * 0.01

        grads = backward(actual_qs, hidden, target_qs, base_model)

        for k, v in grad_buffer.items():
            grads[k] = np.clip(grads[k], -2, 2)
            grad_buffer[k] += grads[k]

        step += 1

        if not step % update_weights_every:
            optimizer.update_params(base_model, grad_buffer)
            grad_buffer = init_grad_buffer(base_model)

        if not step % update_target_net:
            """ After certain timestep set target, online and eval model equal to base model """
            target_model = base_model.copy()
            online_net = base_model.copy()
            eval_net = base_model.copy()

        print(f'epoch: {epoch}, ' +
              f'epsilon: {epsilon}, ' +
              f'smooth rewards: {smooth_rewards:.3f}, ' +
              f'smooth loss: {smooth_loss:.3f}')
