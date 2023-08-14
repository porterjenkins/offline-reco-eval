from typing import Optional, List
import os
import json
import pandas as pd
import datetime

from states import DisplayState
from policy.policies import BasePolicy
from policy.estimators import EnsemblePredictor

class MOPOPolicy(BasePolicy):
    def __init__(self, estimator: EnsemblePredictor, products: Optional[list], rollouts: int = 5, lmbda: float = 1.0):
        super(MOPOPolicy, self).__init__(products)
        self.rollouts = rollouts
        self.estimator = estimator
        self.lmbda = lmbda

    def _get_all_vals(self):
        """
        Return sorted dict of all products and scores
        """
        qvals = {}
        for p in self.products:
            qvals[p] = self.qtable.get(p, 0.0) / self.qcounter.get(p, 1)
        q_sorted = [(k, v) for k, v in sorted(qvals.items(), key=lambda item: item[1], reverse=False)]
        return q_sorted


    def _select_action(self, state: DisplayState, sorted_q_vals: List[tuple]):
        qvals = []
        for p in state.prods:
            qvals.append((p, (self.qtable.get(p, 0) / self.qcounter.get(p, 1))))

        q_sorted = [x for x in sorted(qvals, key=lambda item: item[1], reverse=True)]

        # drop lowest value product
        p_drop = q_sorted.pop()
        budget = state.quantities[p_drop[0]]

        a_prod, a_val = sorted_q_vals.pop()
        a = {a_prod: budget}

        for tup in q_sorted:
            p, v = tup
            a[p] = state.quantities[p]

        return a



    def __call__(self, state: DisplayState):

        #state_vals, state_sigma = self.estimator(state)
        if self.qtable:
            q_vals_all = self._get_all_vals()
            a = self._select_action(state, q_vals_all)
        else:
            a = self.get_random_action(state.max_slots)

        return a


    def update(self, action: dict, payoffs: dict, *args, **kwargs):
        super().update(action, payoffs, *args, **kwargs)
        state = kwargs['rand_state']
        a = {}
        for i in range(self.rollouts):
            if a:
                dummy_state = DisplayState(
                    disp_id=state.disp_id,
                    max_slots=state.max_slots,
                    timestamp=datetime.timedelta(days=i) + state.ts,
                    prod_quantity=a
                )
            else:
                dummy_state = DisplayState(
                    disp_id=state.disp_id,
                    max_slots=state.max_slots,
                    timestamp=datetime.timedelta(days=i) + state.ts,
                    prod_quantity=state.quantities
                )
            cand_vals, cand_sigma = self.estimator(dummy_state)
            q_vals_all = []
            for p, val in cand_vals.items():
                q_vals_all.append(
                    (p, val - self.lmbda * cand_sigma[p])
                )
            q_sorted = [(k, v) for k, v in sorted(q_vals_all, key=lambda item: item[1], reverse=False)]
            a = self._select_action(dummy_state, q_sorted)
            super().update(action=None, payoffs=a)


    @classmethod
    def build_from_dir(cls, model_dir: str, event_data_fpath:str, rollouts:int, lmbda: float):
        predictor = EnsemblePredictor.build_predictor(model_dir)
        dta = pd.read_csv(event_data_fpath)
        policy = MOPOPolicy(
            estimator=predictor,
            products=list(dta['product_id'].unique()),
            rollouts=rollouts,
            lmbda=lmbda

        )
        return policy
