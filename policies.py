import numpy as np


class RandomPolicy(object):

    def __init__(self, products: list, range: tuple):
        self.product = products
        self.q_min, self.q_max = range
        self.n_min, self.n_max = 1, 10


    def __call__(self, state):
        n_prod = np.random.randint(self.n_min, self.n_max)
        prods = np.random.choice(self.product, size=n_prod)
        a = {}
        for p in prods:
            q = np.random.randint(self.q_min, self.q_max)
            a[p] = q
        return a
