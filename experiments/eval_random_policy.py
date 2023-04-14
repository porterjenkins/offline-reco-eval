import pandas as pd

from evaluators import OfflineEvaluator
from policy.policies import RandomPolicy

#fpath = "../data/example-display-2.csv"
fpath = "../data/fall-msd-control.csv"
iter = 100
df = pd.read_csv(fpath)

evaluator = OfflineEvaluator.build_from_csv(
    fpath=fpath,
    iter=iter
)

policy = RandomPolicy(products=list(df['name'].unique()))

evaluator.main(policy=policy)