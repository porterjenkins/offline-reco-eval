import pandas as pd

from evaluators import OfflineEvaluator
from policies import RandomPolicy

fpath = "../data/example-display-2.csv"
df = pd.read_csv(fpath)

evaluator = OfflineEvaluator.build_from_csv(
    fpath=fpath,
    iter=1000
)



policy = RandomPolicy(
    products=list(df['name'].unique()),
    range=(2, 8)
)

evaluator.main(policy=policy)