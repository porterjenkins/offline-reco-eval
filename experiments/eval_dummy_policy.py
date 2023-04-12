import pandas as pd

from evaluators import OfflineEvaluator
from policies import DummyPolicy

#fpath = "../data/example-display-2.csv"
fpath = "../data/fall-msd-treatment.csv"
iter = 1000
eps = 0.0
df = pd.read_csv(fpath)

evaluator = OfflineEvaluator.build_from_csv(
    fpath=fpath,
    iter=iter
)

policy = DummyPolicy()

evaluator.main(policy=policy)