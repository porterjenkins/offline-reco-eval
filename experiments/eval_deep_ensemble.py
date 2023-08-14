import pandas as pd
from evaluators import OfflineEvaluator
from policy.deep_ensemble import EnsemblePredictor, DeepEnsemblePolicy

#fpath = "../data/example-display-2.csv"
fpath = "../data/fall-msd-control_clean.csv"

iter = 30
eps = 0.00
alpha = 0.05
num_swap = 2
event_thin = 0.5
df = pd.read_csv(fpath)

evaluator = OfflineEvaluator.build_from_csv(
    fpath=fpath,
    iter=iter,
    thin=event_thin
)

policy = DeepEnsemblePolicy.build_from_dir(
    model_dir="../models/ensemble",
    eps=eps,
    alpha=alpha,
    num_swap=num_swap,
    event_data_fpath=fpath
)
evaluator.main(policy=policy)