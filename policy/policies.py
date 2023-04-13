import numpy as np
import json
from typing import Optional, List

from states import DisplayState


class BasePolicy(object):

    def __init__(self, products: Optional[list] = None):
        self.n_min, self.n_max = 1, 10
        self.q_min, self.q_max = 2, 8
        self.products = products


    def __call__(self, state: DisplayState):
        # implement by derived class
        pass

    def reset(self):
        # implemented by derived class
        pass

    def update(self, action: dict, payoffs: dict):
        # implemented by derived class
        pass

    def get_random_action(self):
        n_prod = np.random.randint(self.n_min, self.n_max)
        prods = np.random.choice(self.products, size=n_prod)
        a = {}
        for p in prods:
            q = np.random.randint(self.q_min, self.q_max)
            a[p] = q
        return a


class DummyPolicy(BasePolicy):
    """recommend current state (ie, no change)"""
    def __init__(self):
        super(DummyPolicy, self).__init__(products=None)


    def __call__(self, state: DisplayState):
        return state.get_prod_quantites()



class RandomPolicy(BasePolicy):

    def __init__(self, products: list, range: tuple):
        super(RandomPolicy, self).__init__(products=products)
        self.q_min, self.q_max = range
        self.n_min, self.n_max = 1, 10


    def __call__(self, state):
        a = self.get_random_action()
        return a



class EpsilonGreedy(BasePolicy):

    def __init__(self, products: list, eps: float):
        super(BasePolicy, self).__init__(products=products)

        self.eps = eps
        self.qtable = {}
        self.qcounter = {}


    @staticmethod
    def dict_to_str(d: dict):
        return json.dumps(d, sort_keys=True)

    def __call__(self, state: DisplayState):

        alpha = np.random.random()

        if self.qtable and alpha > self.eps:
            qvals = {}
            for k, v in self.qtable.items():
                qvals[k] = v / self.qcounter[k]

            q_sorted = {k: v for k, v in sorted(qvals.items(), key=lambda item: item[1], reverse=True)}
            a = json.loads(list(q_sorted.keys())[0])
        else:
            a = self.get_random_action()

        return a

    def update(self, action: dict, payoffs: dict):
        total_payoff = np.sum(list(payoffs.values()))

        state_dict = self.dict_to_str(action)

        if state_dict not in self.qtable:
            self.qtable[state_dict] = 0
        self.qtable[state_dict] += total_payoff

        if state_dict not in self.qcounter:
            self.qcounter[state_dict] = 0
        self.qcounter[state_dict] += 1


    def reset(self):
        self.qtable = {}
        self.qcounter = {}



class DynamicProgramming(BasePolicy):

    def __init__(self, products: list, max_weight: int = 5):
        super(DynamicProgramming, self).__init__(products=products)
        self.qtable = {}
        self.qcounter = {}
        self.max_weight = max_weight

    def reset(self):
        self.qtable = {}
        self.qcounter = {}

    def update(self, state: DisplayState, payoffs: dict):

        for prod, rew in payoffs.items():
            if prod not in self.qtable:
                self.qtable[prod] = 0
            self.qtable[prod] += rew

            if prod not in self.qcounter:
                self.qcounter[prod] = 0

            self.qcounter[prod] += 1


    def __call__(self, state: DisplayState):
        budget = state.max_slots
        if self.qtable:
            qvals = {}
            for k, v in self.qtable.items():
                qvals[k] = v / self.qcounter[k]

            q_sorted = {k: v for k, v in sorted(qvals.items(), key=lambda item: item[1], reverse=True)}
            a = {}
            for prod, val in q_sorted.items():
                if budget <= 0:
                    break

                if budget < self.max_weight:
                    a[prod] = budget
                    budget -= self.max_weight
                else:
                    a[prod] = self.max_weight
                    budget -= self.max_weight


            #product = list(q_sorted.keys())[0]
            #a = {product: state.max_slots}
        else:
            a = self.get_random_action()

        return a


