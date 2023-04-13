import pandas as pd

from evaluators import OfflineEvaluator
from policy.bayes_3dv import Bayes3dv

#fpath = "../data/example-display-2.csv"
fpath = "../data/fall-msd-treatment.csv"

iter = 100
eps = 0.0
alpha = 0.1
num_swap = 2
df = pd.read_csv(fpath)

evaluator = OfflineEvaluator.build_from_csv(
    fpath=fpath,
    iter=iter
)

policy = Bayes3dv.build_from_dir(
    data_dir="../data",
    event_data_fpath=fpath,
    eps=eps,
    alpha=alpha,
    num_swap=num_swap
)

evaluator.main(policy=policy)