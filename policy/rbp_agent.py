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

