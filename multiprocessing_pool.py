import numpy as np
import multiprocessing as mp
import time
import ray
import gym
from os import getpid

def worker(row):
    rowsum = np.sum(row)
    return rowsum

def get_gym_env():
    pid = getpid()
    observations = []
    total_reward = 0

    env = gym.make('CartPole-v0')
    env.reset()
    done = False
    while not done:
        action = np.random.randint(0, env.action_space.n)
        observation, reward, done, info = env.step(action)
        observations.append(observation)
        total_reward += reward
    # print(f'pid: {pid} | total_reward: {total_reward}')
    return total_reward

if __name__ == '__main__':
    start = time.time()
    total_rewards = []
    for _ in range(10000):
        total_rewards.append(get_gym_env())
    end = time.time()
    print(f'normal processing took: {end - start} seconds')

    start = time.time()
    # Step 1: Init multiprocessing.Pool()
    pool = mp.Pool(mp.cpu_count())

    # Step 2: `pool.apply` the `howmany_within_range()`
    results = [pool.apply(func=get_gym_env, args=()) for _ in range(10000)]

    # Step 3: Don't forget to close
    pool.close()
    pool.join()

    end = time.time()
    print(f'pool processing took: {end - start} seconds')
    print(results)
