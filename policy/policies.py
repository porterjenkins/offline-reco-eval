import numpy as np
import json
import pandas as pd

from states import DisplayState
from policy.rbp_agent import HeuristicAgent, HeuristicCandidateSearch
class BasePolicy(object):

    def __call__(self, state: DisplayState):
        # implement by derived class
        pass

    def reset(self):
        # implemented by derived class
        pass

    def update(self, state: DisplayState, payoffs: dict):
        # implemented by derived class
        pass


class DummyPolicy(BasePolicy):
    """recommend current state (ie, no change)"""
    def __init__(self):
        super(DummyPolicy, self).__init__()


    def __call__(self, state: DisplayState):
        return state.get_prod_quantites()



class RandomPolicy(BasePolicy):

    def __init__(self, products: list, range: tuple):
        super(RandomPolicy, self).__init__()
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



class EpsilonGreedy(BasePolicy):

    def __init__(self, products: list, range: tuple, eps: float):
        super(BasePolicy, self).__init__()
        self.product = products
        self.q_min, self.q_max = range
        self.n_min, self.n_max = 1, 10
        self.eps = eps

        self.qtable = {}
        self.qcounter = {}

    def get_random_action(self):
        n_prod = np.random.randint(self.n_min, self.n_max)
        prods = np.random.choice(self.product, size=n_prod)
        a = {}
        for p in prods:
            q = np.random.randint(self.q_min, self.q_max)
            a[p] = q
        return a

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

    def update(self, state: DisplayState, payoffs: dict):
        total_payoff = np.sum(list(payoffs.values()))

        state_dict = self.dict_to_str(state)

        if state_dict not in self.qtable:
            self.qtable[state_dict] = 0
        self.qtable[state_dict] += total_payoff

        if state_dict not in self.qcounter:
            self.qcounter[state_dict] = 0
        self.qcounter[state_dict] += 1


    def reset(self):
        self.qtable = {}
        self.qcounter = {}



class Bayes3dv(BasePolicy):

    def __init__(self, disp_id, prod_values: dict, prod_std_err: dict, eps: float, alpha: float, num_swap: int):
        self.disp_id = disp_id
        self.prod_values = prod_values
        self.prod_std_err = prod_std_err
        self.eps = eps
        self.alpha = alpha
        self.num_swap = num_swap


    @classmethod
    def build_from_csv(cls, rbp_data_fpath: str, event_data_fpath: str):
        rbp = pd.read_csv(rbp_data_fpath)
        dta = pd.read_csv(event_data_fpath)

        joined = pd.merge(dta, rbp, on=['store_id', 'product_id'], how='left')



    def __call__(self, state: DisplayState):
        a = self.select_action(
            state=state.get_prod_quantites(),
            max_slots=state.max_slots,
        )
        return a

    @staticmethod
    def select_action(
            state: dict,
            vals: dict,
            cands: dict,
            max_slots: int,
            eps: float,
            alpha: float,
            num_swap: int,
    ) -> dict:

        """

        @param state: (dict) previous state of display:
                    - keys are products (str)
                    - values are facing counts (int)
        @param vals: (dict) current expected payoff of each product:
                    - keys are products (str)
                    - values are payoffs (float)
                    - payoff values are expected to be NORMALIZED
        @param cands: (dict) candidate set:
                    - keys are products
                    - values are edge weights
                    - edge weights are expected to be NORMALIZED
        @param max_slots: (int) maximum facings at display
        @param eps: (float) epsilon parameter in epsilon greedy exploration
        @param alpha: (float) alpha parameter controlling the payoff weight in the score function
        @param logger: (Logger) logger object
        @param num_swap: number of products to swap from previous state
        @return: (dict) facings to allocate

        The action selection follows the following general logic:
            - candidates by their values
            - sort the current state by the product values
            - remove the bottom num_swap products
                - with probability epsilon randomly select a candidate. Probabilites are weights by values
                - with probability 1-epsilon greedily search through candidates by values
            - if the number allocated is less than the budget (max_slots), do another greedy search to allocate all
            - return action dictionary
        """
        # merge candidates and values
        cand_vals = {}
        val_mean = np.mean(list(vals.values()))
        for c, edge in cands.items():
            if c not in vals:
                v = val_mean
            else:
                v = vals[c]

            cand_vals[c] = HeuristicAgent.get_weighted_payoff_score(alpha, v, edge)

        # sort payoff values: Descending
        cand_vals = {k: v for k, v in sorted(cand_vals.items(), key=lambda item: item[1], reverse=True)}

        state_vals = {k: vals.get(k, 0.0) for k, v in state.items()}
        state_vals = {k: v for k, v in sorted(state_vals.items(), key=lambda item: item[1])}


        action = {}

        allocated = 0
        cntr = 0
        heuristic = HeuristicCandidateSearch(vals=cand_vals, presorted=True)
        for p_id, p_val in state_vals.items():
            if cntr < num_swap:
                alpha = np.random.random()
                if alpha < eps:
                    p_prime = heuristic.choice(action, state, max_iter=20)
                else:
                    p_prime = heuristic.greedy_search(action, state)
                q = int(state[p_id])
                q_prime, remain = HeuristicAgent.decay_quantity(q)

                if p_prime is not None:
                    action[p_prime] = q_prime
                    allocated += q_prime
                    if remain > 0:
                        action[p_id] = remain
                        allocated += remain
            else:
                q = int(state[p_id])
                action[p_id] = q
                allocated += q

            cntr += 1

        while allocated < max_slots:
            p_prime = heuristic.greedy_search(action, state)
            budget = max_slots - allocated
            q_prime, remain = HeuristicAgent.decay_quantity(budget=budget)
            if p_prime is None:
                # reached end of candidate list
                break
            else:
                action[p_prime] = q_prime
                allocated += q_prime


        return action

