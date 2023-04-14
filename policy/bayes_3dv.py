import numpy as np
import pandas as pd
import os
import json
import math

from policy.policies import BasePolicy
from policy.rbp_agent import HeuristicCandidateSearch
from policy.candidates import CandidateGenerator
from states import DisplayState


class Bayes3dv(BasePolicy):

    def __init__(
            self,
            rbp_vals: dict,
            rbp_std: dict,
            adj_list: dict,
            eps: float,
            alpha: float,
            num_swap: int,
            products: list
    ):
        super(Bayes3dv, self).__init__(products)
        self.rbp_vals = rbp_vals
        self.rbp_std = rbp_std
        self.eps = eps
        self.alpha = alpha
        self.num_swap = num_swap
        self.adj_list = adj_list


    @classmethod
    def build_from_dir(cls, data_dir: str, event_data_fpath, eps: float, alpha:float, num_swap: int):
        rbp_data_fpath = os.path.join(data_dir, "bigquery_rbp_run.csv")
        adj_fpath = os.path.join(data_dir, "adj_list.json")
        lmbda = 1.0

        rbp = pd.read_csv(rbp_data_fpath)
        dta = pd.read_csv(event_data_fpath)

        with open(adj_fpath, "r") as f:
            adj = json.load(f)

        joined = pd.merge(dta, rbp, on=['store_id', 'product_id'], how='inner')
        print(f"Transactions dropped: {dta.shape[0] - joined.shape[0]}")

        rbps = joined[['display_id', 'product_id', 'posterior_mean', 'posterior_std']].groupby(
            ['display_id', 'product_id']).max().reset_index()

        rbp_vals = {}
        rbp_stds = {}

        for disp, vals in rbps.groupby("display_id"):
            #rbp_vals[disp] = dict(zip(vals['product_id'], vals['posterior_mean']))
            rbp_stds[disp] = dict(zip(vals['product_id'], vals['posterior_std']))
            penalized = vals['posterior_mean'] - lmbda*vals['posterior_std']
            rbp_vals[disp] = dict(zip(vals['product_id'], penalized))

        b3dv_policy = Bayes3dv(
            rbp_vals=rbp_vals,
            rbp_std=rbp_stds,
            adj_list=adj,
            eps=eps,
            alpha=alpha,
            num_swap=num_swap,
            products=list(dta['name'].unique())
        )

        return b3dv_policy


    def _generate_candidates(self, state_dict: dict):
        generator = CandidateGenerator()
        #state_dict = self.get_disp_state(disp_id)
        current_product_list = list(state_dict.keys())

        #acceptable_products = DatasetManager.store_dist_dict["company_products"]
        #if store_id in DatasetManager.store_dist_dict.keys():
        #    acceptable_products.append(DatasetManager.store_dist_dict[store_id])

        #for product_id in DatasetManager.forbidden_products:
        #    if product_id in acceptable_products:
        #        acceptable_products.remove(product_id)

        current_product_list = list(state_dict.keys())
        cand_set = generator.gen(self.adj_list, current_product_list)

        return cand_set


    def __call__(self, state: DisplayState):
        disp_id = state.disp_id
        if disp_id in self.rbp_vals:
            cand_set = self._generate_candidates(state.get_prod_quantites())
            a = self.select_action(
                state=state.get_prod_quantites(),
                max_slots=state.max_slots,
                vals = self.rbp_vals[disp_id],
                eps=self.eps,
                alpha=self.alpha,
                num_swap=self.num_swap,
                cands=cand_set
            )
        else:
            a = self.get_random_action(state.max_slots)
        return a


    @staticmethod
    def decay_quantity(budget):
        if budget <= 2:
            q = budget
        else:
            # allocate candidate product with half of remaining budget
            q = math.ceil(budget / 2)
        diff = budget - q
        return q, diff

    @staticmethod
    def get_weighted_payoff_score(alpha: float, val: float, edge: float):
        """
        Compute a weighted payoff estimate:
            payoff = alpha*value + (1-alpha)*edge
        @param alpha: (float) weighting factor given the value:
                            -   complement is given to weight (1-alpha)
                            - Must be in [0, 1]
        @param val: (float) expected payoff
        @param edge: (float) candidate edge weight
        @return: (float) weighted candidate score
        """

        if alpha < 0 or alpha > 1:
            raise ValueError(f"alpha must b in [0, 1]. Got {alpha}")

        return alpha * val + (1 - alpha) * edge

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

            cand_vals[c] = Bayes3dv.get_weighted_payoff_score(alpha, v, edge)

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
                q_prime, remain = Bayes3dv.decay_quantity(q)

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
            q_prime, remain = Bayes3dv.decay_quantity(budget=budget)
            if p_prime is None:
                # reached end of candidate list
                break
            else:
                action[p_prime] = q_prime
                allocated += q_prime


        return action
