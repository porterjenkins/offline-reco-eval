import pandas as pd

from evaluators import OfflineEvaluator
from policy.ablation import *


fpath = "../data/fall-msd-control_clean.csv"

iter = 30
eps = 0.00
alpha = 0.05
num_swap = 2
event_thin = 0.5
df = pd.read_csv(fpath)

evaluator = OfflineEvaluator.build_from_csv(
    fpath=fpath,
    iter=iter,
    thin=event_thin
)

policy = NoSearch.build_from_dir(
    data_dir="../data",
    event_data_fpath=fpath,
    eps=eps,
    alpha=alpha,
    num_swap=num_swap
)

evaluator.main(policy=policy)