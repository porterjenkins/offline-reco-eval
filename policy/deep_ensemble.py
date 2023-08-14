import json
import os
import numpy as np
import pandas as pd

from policy.estimators import EnsemblePredictor
from states import DisplayState

from policy.bayes_3dv import Bayes3dv


class DeepEnsemblePolicy(Bayes3dv):

    def __init__(self, estimator: EnsemblePredictor, adj_list: dict, eps: float, alpha: float, num_swap: int, products: list):
        super(DeepEnsemblePolicy, self).__init__(
            adj_list=adj_list,
            eps=eps,
            alpha=alpha,
            num_swap=num_swap,
            products=products,
            rbp_vals=None,
            rbp_std=None
        )
        self.estimator = estimator


    def __call__(self, state: DisplayState, *args, **kwargs):


        state_vals, state_sigma = self.estimator(state)
        cand_set = self._generate_candidates(state.get_prod_quantites())
        if cand_set:
            dummy_state = DisplayState(
                disp_id=state.disp_id,
                max_slots=state.max_slots,
                timestamp=state.ts,
                prod_quantity=dict(zip(cand_set.keys(), np.ones(len(cand_set))))
            )
            cand_vals, cand_sigma = self.estimator(dummy_state)
        else:
            num_cands = 20
            cands = np.random.choice(self.products, num_cands, replace=False)
            cand_vals = dict(
                zip(
                    cands,
                    np.ones(num_cands) * np.mean(list(state_vals.values()))
                )
            )
            cand_sigma = dict(
                zip(
                    cands,
                    np.ones(num_cands) * np.mean(list(state_sigma.values()))
                )
            )
        state_vals.update(cand_vals)
        state_sigma.update(cand_sigma)

        """state_vals_discounted = {}
        for k, v in state_vals.items():
            state_vals_discounted[k] = v - 2.0*state_sigma[k]"""

        a = self.select_action(
            state=state.get_prod_quantites(),
            max_slots=state.max_slots,
            vals=state_vals,
            eps=self.eps,
            alpha=self.alpha,
            num_swap=self.num_swap,
            cands=cand_set
        )
        return a


    @classmethod
    def build_from_dir(cls, model_dir: str, event_data_fpath:str, eps: float, alpha:float, num_swap: int):
        predictor = EnsemblePredictor.build_predictor(model_dir)
        adj_fpath = os.path.join(model_dir, "adj_list.json")
        with open(adj_fpath, "r") as f:
            adj = json.load(f)

        dta = pd.read_csv(event_data_fpath)

        policy = DeepEnsemblePolicy(
            estimator=predictor,
            adj_list=adj,
            eps=eps,
            alpha=alpha,
            num_swap=num_swap,
            products=list(dta['product_id'].unique())
        )
        return policy
