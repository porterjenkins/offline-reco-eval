import pandas as pd

from evaluators import OfflineEvaluator
from policy.policies import LinearProgramming

#fpath = "../data/example-display-2.csv"
fpath = "../data/fall-msd-treatment.csv"
iter = 10
eps = 0.1
df = pd.read_csv(fpath)

evaluator = OfflineEvaluator.build_from_csv(
    fpath=fpath,
    iter=iter
)

policy = LinearProgramming(
    products=list(df['name'].unique()),
    max_weight=5
)

evaluator.main(policy=policy)