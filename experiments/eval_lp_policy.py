import pandas as pd

from evaluators import OfflineEvaluator
from policy.policies import LinearProgramming

#fpath = "../data/example-display-2.csv"
fpath = "../data/fall-msd-control.csv"
iter = 100
eps = 0.1
max_w = 8
df = pd.read_csv(fpath)

evaluator = OfflineEvaluator.build_from_csv(
    fpath=fpath,
    iter=iter
)

policy = LinearProgramming(
    products=list(df['product_id'].unique()),
    max_weight=max_w
)

evaluator.main(policy=policy)