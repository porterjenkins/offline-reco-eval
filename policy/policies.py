import numpy as np
import json
from typing import Optional, List

from scipy.optimize import linprog

from states import DisplayState


class BasePolicy(object):

    def __init__(self, products: Optional[list] = None):
        self.n_min, self.n_max = 1, 10
        self.q_min, self.q_max = 0, 8
        self.products = products
        self.qtable = {}
        self.qcounter = {}


    def __call__(self, state: DisplayState):
        # implement by derived class
        pass


    def reset(self):
        self.qtable = {}
        self.qcounter = {}

    def get_random_action(self, max_slots):

        budget = max_slots
        a = {}
        while budget > 0:
            p = np.random.choice(self.products, size=1)[0]
            q = np.random.randint(self.q_min, min(self.q_max, budget)+1)
            a[p] = a
            budget -= q

        return a

    def update(self, action: dict, payoffs: dict, *args, **kwargs):

        for prod, rew in payoffs.items():
            if prod not in self.qtable:
                self.qtable[prod] = 0
            self.qtable[prod] += rew

            if prod not in self.qcounter:
                self.qcounter[prod] = 0

            self.qcounter[prod] += 1


class DummyPolicy(BasePolicy):
    """recommend current state (ie, no change)"""
    def __init__(self):
        super(DummyPolicy, self).__init__(products=None)


    def __call__(self, state: DisplayState):
        return state.get_prod_quantites()



class RandomPolicy(BasePolicy):

    def __init__(self, products: list):
        super(RandomPolicy, self).__init__(products=products)


    def __call__(self, state):
        a = self.get_random_action(state.max_slots)
        return a


class GeneticPolicy(BasePolicy):

    def __init__(self, products: list, marriage_rate: float):
        super(GeneticPolicy, self).__init__(products=products)
        self.marriage_rate = marriage_rate

    def reset(self):
        pass

    @staticmethod
    def dict_to_str(d: dict):
        return json.dumps(d, sort_keys=True)

    def update(self, state: DisplayState, payoffs: dict, *args, **kwargs):
        total_payoff = np.sum(list(payoffs.values()))

        state_dict = self.dict_to_str(state)

        if state_dict not in self.qtable:
            self.qtable[state_dict] = 0
        self.qtable[state_dict] += total_payoff

        if state_dict not in self.qcounter:
            self.qcounter[state_dict] = 0
        self.qcounter[state_dict] += 1

    def marry_partners(self, original_partner, partner_2):
        #Sort Alphabetically
        original_partner = {k: v for k, v in sorted(original_partner.items(), key=lambda item: item[0])}
        partner_2 = {k: v for k, v in sorted(partner_2.items(), key=lambda item: item[0])}

        #Convert to lists
        original_partner_list = []
        for k, v in original_partner.items():
            for _ in range(int(v)):
                original_partner_list.append(k)

        partner_2_list = []
        for k, v in partner_2.items():
            for _ in range(int(v)):
                partner_2_list.append(k)
        
        

        splice_point = round(np.random.random() * (len(original_partner_list)))

        if splice_point < len(original_partner_list):
            for i in range(splice_point, min(len(original_partner_list), len(partner_2_list))):
                original_partner_list[i] = partner_2_list[i]

        # Convert to dict
        child_dict = {}
        for id in original_partner_list:
            if id in child_dict:
                child_dict[id] += 1
            else:
                child_dict[id] = 1

        return child_dict

    def __call__(self, state: DisplayState):
        alpha = np.random.random()
        if self.qtable and alpha < self.marriage_rate:
            qvals = {}
            for k, v in self.qtable.items():
                qvals[k] = v / self.qcounter[k]

            q_sorted = {k: v for k, v in sorted(qvals.items(), key=lambda item: item[1], reverse=True)}

            partner_1 = state.quantities
            partner_2 = json.loads(list(q_sorted.keys())[0])

            # print("MARRIAGE")
            a = self.marry_partners(partner_1, partner_2)
        else:
            # print("MUTATION")
            a = self.get_random_action(state.max_slots)

        return a

class EpsilonGreedy(BasePolicy):

    def __init__(self, products: list, eps: float):
        super(EpsilonGreedy, self).__init__(products=products)

        self.eps = eps


    @staticmethod
    def dict_to_str(d: dict):
        return json.dumps(d, sort_keys=True)

    def update(self, state: DisplayState, payoffs: dict, *args, **kwargs):
        total_payoff = np.sum(list(payoffs.values()))

        state_dict = self.dict_to_str(state)

        if state_dict not in self.qtable:
            self.qtable[state_dict] = 0
        self.qtable[state_dict] += total_payoff

        if state_dict not in self.qcounter:
            self.qcounter[state_dict] = 0
        self.qcounter[state_dict] += 1

    def __call__(self, state: DisplayState):

        alpha = np.random.random()

        if self.qtable and alpha > self.eps:
            qvals = {}
            for k, v in self.qtable.items():
                qvals[k] = v / self.qcounter[k]

            q_sorted = {k: v for k, v in sorted(qvals.items(), key=lambda item: item[1], reverse=True)}
            a = json.loads(list(q_sorted.keys())[0])
        else:
            a = self.get_random_action(state.max_slots)

        return a






class DynamicProgramming(BasePolicy):

    def __init__(self, products: list, max_weight: int = 5):
        super(DynamicProgramming, self).__init__(products=products)
        self.qtable = {}
        self.qcounter = {}
        self.max_weight = max_weight


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
            a = self.get_random_action(state.max_slots)

        return a


class LinearProgramming(BasePolicy):

    def __init__(self, products: list, max_weight: int):
        super(LinearProgramming, self).__init__(products=products)
        self.n_products = len(products)
        self.p_to_idx = dict(zip(products, range(self.n_products)))
        self.idx_to_p = dict(zip(range(self.n_products), products))
        self.lin_prog_coeff = np.zeros(self.n_products)
        self.max_weight = max_weight


    def reset(self):
        BasePolicy.reset(self)
        self.lin_prog_coeff = np.zeros(self.n_products)

    def update(self, action: dict, payoffs: dict, *args, **kwargs):
        BasePolicy.update(self, action, payoffs)
        for k, v in self.qtable.items():
            idx = self.p_to_idx[k]
            c = self.qtable[k] / self.qcounter[k]
            self.lin_prog_coeff[idx] = c
    def __call__(self, state: DisplayState):

        # coeff vector
        coef = -1*self.lin_prog_coeff
        # equality constraints: facings sum to max_slots
        A_eq = np.ones(self.n_products).reshape(1, -1)
        b_eq = np.array([state.max_slots])
        # inequality constraints
        A_iq = np.eye(self.n_products)
        b_iq = np.ones(self.n_products) * self.max_weight

        # bounds on variables
        bounds = [[0, state.max_slots] for i in range(self.n_products)]
        try:
            res = linprog(coef, A_eq=A_eq, b_eq=b_eq, A_ub=A_iq, b_ub=b_iq, bounds=bounds)
        except ValueError:
            stop = 0
        x_star = res.x
        a = {}
        for i, x_i in enumerate(x_star):
            if x_i > 0:
                a[self.idx_to_p[i]] = x_i

        return a

