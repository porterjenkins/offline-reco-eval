import pandas as pd

from evaluators import OfflineEvaluator
from policy.policies import Bayes3dv

#fpath = "../data/example-display-2.csv"
fpath = "../data/fall-msd-treatment.csv"
rbp_data_fpath = "../data/bigquery_rbp_run.csv"
iter = 100
eps = 0.1
df = pd.read_csv(fpath)

evaluator = OfflineEvaluator.build_from_csv(
    fpath=fpath,
    iter=iter
)

policy = Bayes3dv.build_from_csv(
    event_data_fpath=fpath,
    rbp_data_fpath=rbp_data_fpath
)

evaluator.main(policy=policy)