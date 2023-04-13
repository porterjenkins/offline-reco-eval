import pandas as pd
import json
import numpy as np
import os
import random
import math
from typing import Optional, Dict, List, Tuple
from logging import Logger

#from reco.agents.base_agent import BaseRecoAgent
#from reco.exceptions import CandidateSetEmpty
#from reco.reco_utils import log_dict, log_iter
#from reco.dataset_manager import DatasetManager, GroupedDataManger
#from reco.candidate_generator import CandidateGenerator

from sklearn.linear_model import LinearRegression

class BaseRecoAgent(object):

    reward_col_name = 'reward'

    def __init__(self, group_id: str, logger: Optional[Logger] = None):
        self.group_id = group_id
        self.id = uuid4()
        self.dta = None
        self.logger = logger


    def _log_info_msg(self, msg):

        if self.logger:
            if isinstance(msg, dict):
                log_dict(self.logger, msg)
            else:
                self.logger.info(msg)


    def _get_reward(self, dta):
        """
        Compute the time (in days) adjusted reward:
            reward = (quantity_sold * price) / time

        @param dta: (pd.Dataframe)
        @return: (pd.Series)
        """

        reward = dta['quantity_sold'] / dta['timedelta']
        return reward


    def _get_preprocessed_data(self, dta: GroupedDataManger):
        if len(dta.obs) == 0:
            return dta

        dta.add_col(
            self._get_reward(
                dta.get_obs()
            ),
            name=self.reward_col_name
        )

        dta.preprocess(outlier_col_name=self.reward_col_name)


        return dta

    def _get_validated_display_states(self, dta: GroupedDataManger):
        return dta.get_validated_display_state()

    def train(self):
        # implemented by derived class
        pass

    def predict(self):
        # implemented by derived class
        pass

    @staticmethod
    def select_action(**kwargs):
        # implemented by derived class
        pass

    @staticmethod
    def get_yaml(fpath):
        with open(fpath, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        return cfg

    @classmethod
    def build_from_yaml(cls, fname, display_id):
        # implemented by derived class
        pass


class HeuristicCandidateSearch(object):
    def __init__(self, vals: dict, presorted: bool):
        if not presorted:
            # sort payoff values: Descending
            vals = {k: v for k, v in sorted(vals.items(), key=lambda item: item[1], reverse=True)}
        self.sorted_cand_list = list(zip(vals.keys(), vals.values()))
        self.cands = list(vals.keys())
        self.vals = np.array(list(vals.values()))
        self.probs = self.vals / self.vals.sum()

        self.query_idx = 0

    def greedy_search(self, action, state):
        search = True
        p_prime = None
        while search:
            # search for candidate from sorted list
            # candidate cannot be in current state
            # candidate cannot be previously allocated
            if self.query_idx >= len(self.sorted_cand_list):
                # no candidate found
                p_prime = None
                break
            p_prime, _ = self.sorted_cand_list[self.query_idx]
            self.query_idx += 1
            if p_prime not in action and p_prime not in state:
                search = False

        return p_prime

    def choice(self, action, state, max_iter):
        search = True
        p = None
        cntr = 0
        while search:
            if cntr >= max_iter:
                break
            p_prime = np.random.choice(self.cands, size=1, replace=False, p=self.probs)[0]
            if p_prime not in action and p_prime not in state:
                p = p_prime
                search = False
            cntr += 1
        return p



class HeuristicAgent(BaseRecoAgent):

    cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "cfgs/heuristic.yaml"
    )

    constant_score = 1.0

    def __init__(
            self,
            group_id: str,
            dta: GroupedDataManger,
            eps: float,
            alpha: float,
            num_swap: int,
            logger: Optional[Logger] = None
    ):
        super(HeuristicAgent, self).__init__(
            group_id,
            logger=logger
        )
        self.eps=eps
        self.disp_states = self._get_validated_display_states(dta)
        self.dta = self._get_preprocessed_data(dta)

        # We must run _get_validated_display_states before we run _get_preprocess_data (for obvious reasons)
        # If running _get_preprocessed_data filters enough data to not bring the number of scans below the min
        # threshold, we throw an error and update that information in the validation manager. However, here we need
        # to come back and filter those displays out.
        unsuccessful_display_validation = set(self.disp_states.keys()) - set(self.dta.validation_manager.successful_ids)
        for k in unsuccessful_display_validation:
            self.disp_states.pop(k, None)

        self.alpha = alpha
        self.num_swap = num_swap


    def _get_vals(self, obs: pd.DataFrame):
        # implemented by derived class
        pass

    def train(self):
        """
        Get average of observed trajectories
        @return: (dict) keys: products, values: product-value estimates
        """
        self._log_info_msg(f"TRAINING agent: {self.id}, group: {self.group_id}")
        obs = self.dta.get_obs()
        val_est = self._get_vals(obs)

        self._log_info_msg(f"Value Estimate:")
        log_dict(self.logger, val_est.sort_values(by=self.reward_col_name, ascending=False).to_dict()[self.reward_col_name])

        self.val_est = val_est

    @staticmethod
    def get_probs(cands: list, weights: dict, logger: Logger):

        weights_filtered = []
        for c in cands:
            weights_filtered.append(weights.get(c, 0.0))

        weights_filtered = np.array(weights_filtered)
        probs = weights_filtered / weights_filtered.sum()

        logger.info(f"Candidate probs: {probs}")
        return probs


    @staticmethod
    def randomize_action(
            action: dict,
            cands: list,
            weights: dict,
            min_cnt: int,
            min_c: str,
            logger: Optional[Logger] = None
    ):
        """
        Used for epsilon-greedy selection. Conservative randomization of greedy action by:

        @param action:
        @param list:
        @param min_cnt:
        @param min_c:
        @return: (dict) updated action with random pertubation
        """
        if len(action) == 1:
            # if single product then don't randomize
            logger.info("Only a single product in action set: don't randomize")
            return action
        if not cands:
            # no available candidates
            logger.info("No available candidates: use greedy action")
            return action


        probs = HeuristicAgent.get_probs(cands, weights, logger)

        rand_cand = np.random.choice(cands, p=probs)
        if logger:
            logger.info(f"Removing product: {min_c} count {min_cnt}")

        # drop smallest recomendation
        del action[min_c]

        # re-allocate to randomly sampled candidate
        if not rand_cand in action:
            action[rand_cand] = 0
        action[rand_cand] += min_cnt

        if logger:
            logger.info(f"Adding product: {rand_cand} count {min_cnt}")

        return action


    @staticmethod
    def get_set_diff(action_set: set, cand_set: set):
        """
        Get set difference of candidate set and action set:
            - sample products that are not currently in greedy action

        @param action_set: set of products from greedy action
        @param cand_set: set of products in total candidate set:
                            - action set is a subset of this
        @return: (list)
        """
        cands_prime = cand_set - action_set
        return list(cands_prime)

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

        if val < 0 or val > 1:
            raise ValueError(f"alpha must b in [0, 1]. Got {alpha}")

        return alpha*val + (1-alpha)*edge

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
    def select_action(
            state: dict,
            vals: dict,
            cands: dict,
            max_slots: int,
            eps: float,
            alpha: float,
            num_swap: int,
            logger: Optional[Logger] = None
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

        if logger:
            logger.info("State values")
            log_dict(logger, state_vals)

        action = {}

        allocated = 0
        cntr = 0
        heuristic = HeuristicCandidateSearch(vals=cand_vals, presorted=True)
        if logger:
            logger.info("Action Selection:")
        for p_id, p_val in state_vals.items():
            if cntr < num_swap:
                alpha = np.random.random()
                if alpha < eps:
                    if logger:
                        logger.info("Exploration action")
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
                    if logger:
                        logger.info(f"Swap: {p_id} --> {p_prime}")
                        logger.info(f"Quantity: {q} --> ({remain}, {q_prime})")
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

        if logger:
            logger.info("Current State")
            log_dict(logger, state)
            logger.info("Proposed State")
            log_dict(logger, action)

            add = set(action.keys()) - set(state.keys())
            drop = set(state.keys()) - set(action.keys())
            if add:
                logger.info("Products Added:")
                log_iter(logger, add)
            if drop:
                logger.info("Products Dropped:")
                log_iter(logger, drop)


        return action


    def get_disp_state(self, disp_id: str):
        return self.disp_states[disp_id]

    def get_validated_disp_ids(self):
        return list(self.disp_states.keys())

    def _get_validated_displays_and_stores(self):
        obs = self.dta.get_obs()
        obs = obs[obs["display_id"].isin(self.get_validated_disp_ids())]
        tuples = obs[["display_id", "store_id"]].drop_duplicates().itertuples(index=False, name=None)
        return list(tuples)


    def predict(self):
        """
        return dict schema:
            "action": (dict) dictionary of product, slot counts
            "score: (float) ranking score of the recommendation. Used for sorting downstream
        @return:
        """
        self._log_info_msg(f"PREDICT: agent: {self.id}, display: {self.display_id}")

        target_name = self.reward_col_name + "_normalized"
        self.val_est[target_name] = self.val_est[self.reward_col_name] / self.val_est[self.reward_col_name].sum()
        probs = self.val_est.to_dict()[target_name]

        cand_set = self.dta.get_cands()
        num_slots = self.dta.max_slots
        weights = self.dta.get_weights()

        reco = {
            "action": self.select_action(probs, cand_set, weights, num_slots, self.eps, self.logger),
            "score": self.constant_score
        }

        return reco



class LinearHeuristicAgent(HeuristicAgent):
    cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "cfgs/linear_heuristic.yaml"
    )

    def __init__(self, group_id: str, dta: GroupedDataManger, eps: float, alpha: float, num_swap: int, logger: Logger):
        super(LinearHeuristicAgent, self).__init__(
            group_id=group_id,
            eps=eps,
            logger=logger,
            dta=dta,
            alpha=alpha,
            num_swap=num_swap
        )

    def predict(self):
        """
        return dict schema:
            "action": (dict) dictionary of product, slot counts
            "score: (float) ranking score of the recommendation. Used for sorting downstream
        @return:
        """
        self._log_info_msg(f"PREDICT: agent: {self.id}, Group: {self.group_id}")

        target_name = self.reward_col_name + "_normalized"
        self.val_est[target_name] = self.val_est[self.reward_col_name] / self.val_est[self.reward_col_name].sum()
        vals = self.val_est.to_dict()[target_name]

        # stratify action selection by display
        tuples = self._get_validated_displays_and_stores()
        if len(tuples) == 0:
            return []

        disp_ids, store_ids = zip(*tuples)
        generator = CandidateGenerator(self.logger)

        disp_reco = []
        for disp_id, store_id in zip(disp_ids, store_ids):

            self._log_info_msg(f"Display: {disp_id}")
            state_dict = self.get_disp_state(disp_id)
            current_product_list = list(state_dict.keys())

            acceptable_products = DatasetManager.store_dist_dict["company_products"]
            if store_id in DatasetManager.store_dist_dict.keys():
                acceptable_products.append(DatasetManager.store_dist_dict[store_id])

            for product_id in DatasetManager.forbidden_products:
                if product_id in acceptable_products:
                    acceptable_products.remove(product_id)

            cand_set = generator.gen(DatasetManager.adj_list, acceptable_products, current_product_list)
            num_slots = self.dta.get_max_slot(disp_id)

            if len(cand_set) == 0:
                self.dta.validation_manager.add_exception(CandidateSetEmpty(disp_id))
                continue

            reco = {
                "action": self.select_action(
                    state=state_dict,
                    vals=vals,
                    cands=cand_set,
                    max_slots=num_slots,
                    logger=self.logger,
                    eps=self.eps,
                    alpha=self.alpha,
                    num_swap=self.num_swap
                ),
                "score": self.constant_score,
                "display_id": disp_id
            }

            disp_reco.append(reco)

        return disp_reco

    def _get_vals(self, obs: pd.DataFrame) -> pd.DataFrame:
        """
        Get product-level value estimates
        @param obs: (pd.DataFrame) observed data
        @return: (Dict) Dictionary: keys are products, values are weights
        """
        obs.to_csv(f"./output/{self.group_id}-vals.csv")
        val_est = {}
        for prod, dta in obs.groupby("product_id"):
            linear_model = LinearRegression(fit_intercept=False)
            X = dta["previous_post_scan_num_facings"].values.reshape(-1, 1)
            y = dta[self.reward_col_name].values.reshape(-1, 1)
            with open('data_reco/pid_to_name.json') as json_file:
                product_dict = json.load(json_file)
                prod_name = product_dict.get(prod, prod)
                prod_name = "".join(x for x in prod_name if x.isalnum())
                dta[['store_name', 'cluster_id', 'display_type', 'display_id', 'name', "product_id", 'upc', 'datetime', 'timedelta','previous_post_scan_num_facings','quantity_sold', 'reward']].to_csv(f"./product_output/{prod_name}-{prod}.csv")
            try:
                linear_model.fit(X, y)
                val_est[prod] = linear_model.coef_[0][0]
            except Exception as e:
                self.logger.warning(f'Error prediction value estimate for product {prod}: {e}')
                val_est[prod] = 0

        for product_id in DatasetManager.forbidden_products:
            if product_id in val_est.keys():
                val_est[product_id] = 0

        val_df = pd.DataFrame.from_dict(val_est, orient='index', columns=[self.reward_col_name])
        return val_df