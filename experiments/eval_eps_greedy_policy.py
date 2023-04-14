import pandas as pd

from evaluators import OfflineEvaluator
from policy.policies import EpsilonGreedy

#fpath = "../data/example-display-2.csv"
fpath = "../data/fall-msd-control.csv"
iter = 100
eps = 0.1
event_thin = 0.5
df = pd.read_csv(fpath)

evaluator = OfflineEvaluator.build_from_csv(
    fpath=fpath,
    iter=iter,
    thin=event_thin
)

policy = EpsilonGreedy(
    products=list(df['name'].unique()),
    eps=eps
)

evaluator.main(policy=policy)