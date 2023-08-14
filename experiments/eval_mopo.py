import pandas as pd

from evaluators import OfflineEvaluator
from policy.mopo import MOPOPolicy

#fpath = "../data/example-display-2.csv"
fpath = "../data/fall-msd-control_clean.csv"
iter = 30
event_thin = 0.5
rollouts = 5
lmbda = 0.1
df = pd.read_csv(fpath)

evaluator = OfflineEvaluator.build_from_csv(
    fpath=fpath,
    iter=iter,
    thin=event_thin
)


policy = MOPOPolicy.build_from_dir(
    model_dir="../models/ensemble",
    event_data_fpath=fpath,
    rollouts=rollouts,
    lmbda=lmbda
)

evaluator.main(policy=policy)