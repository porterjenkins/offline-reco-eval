import pandas as pd

from evaluators import OfflineEvaluator
from policies import EpsilonGreedy

#fpath = "../data/example-display-2.csv"
fpath = "../data/fall-msd-treatment.csv"
iter = 100
eps = 0.0
df = pd.read_csv(fpath)

evaluator = OfflineEvaluator.build_from_csv(
    fpath=fpath,
    iter=iter
)

policy = EpsilonGreedy(
    products=list(df['name'].unique()),
    range=(2, 8),
    eps=eps
)

evaluator.main(policy=policy)