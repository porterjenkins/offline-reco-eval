import pandas as pd

from evaluators import OfflineEvaluator
from policy.policies import GeneticPolicy, RandomPolicy

fpath = "../data/fall-msd-control.csv"
iter = 100
event_thin = 0.5
df = pd.read_csv(fpath)

evaluator = OfflineEvaluator.build_from_csv(
    fpath=fpath,
    iter=iter,
    thin=event_thin
)

policy = GeneticPolicy(products=list(df['name'].unique()), marriage_rate=0.7)

evaluator.main(policy=policy)