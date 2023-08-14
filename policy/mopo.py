from typing import Optional
import os
import json
import pandas as pd

from states import DisplayState
from policy.policies import BasePolicy
from policy.estimators import EnsemblePredictor

class MOPOPolicy(BasePolicy):
    def __init__(self, estimator: EnsemblePredictor, products: Optional[list], max_weight: int = 5):
        super(MOPOPolicy, self).__init__(products)
        self.max_weight = max_weight
        self.estimator = estimator


    def __call__(self, state: DisplayState):

        #state_vals, state_sigma = self.estimator(state)
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

    @classmethod
    def build_from_dir(cls, model_dir: str, event_data_fpath:str):
        predictor = EnsemblePredictor.build_predictor(model_dir)
        dta = pd.read_csv(event_data_fpath)
        policy = MOPOPolicy(
            estimator=predictor,
            products=list(dta['product_id'].unique())
        )
        return policy
