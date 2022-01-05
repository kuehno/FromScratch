import numpy as np
import multiprocessing as mp
import time
import ray
import gym
from os import getpid

@ray.remote
def worker(row):
    rowsum = np.sum(row)
    return rowsum

def worker_2(row):
    rowsum = np.sum(row)
    return rowsum

if __name__ == '__main__':
    ray.init()

    big_array = np.random.randint(0, 10, size=[200000, 5])

    start = time.time()
    results = []
    for row in big_array:
        results.append(worker_2(row))
    end = time.time()
    print(f'normal processing took: {end - start} seconds')

    start = time.time()

    futures = [worker.remote(row) for row in big_array]
    print(ray.get(futures))

    end = time.time()
    print(f'pool processing took: {end - start} seconds')
    print(results)
