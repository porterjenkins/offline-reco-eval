import pandas as pd
from sklearn.linear_model import LinearRegression

fpath = "./msd_clustered_2024-01-30.csv"
min_sample_tol = 5

msd = pd.read_csv(fpath)
msd = msd[~pd.isnull(msd['num_facings'])]
print(msd.columns)

global_reg_output = []

for k, dta in msd[['product_id', 'segment_id', 'store_id', 'payoff', 'num_facings']].groupby(
        ["segment_id", "store_id", "product_id"]
):
    seg, store, prod = k
    if dta.shape[0] < min_sample_tol:
        continue
    reg = LinearRegression()
    X = dta['num_facings'].values.reshape(-1,1)
    y = dta['payoff'].values.reshape(-1,1)

    reg.fit(X, y)
    print(seg, store, prod, reg.coef_[0][0])
    global_reg_output.append(
        [seg, store, prod,reg.coef_[0][0], 0.0]
    )


global_reg_output = pd.DataFrame(
    global_reg_output,
    columns=["segment_id", "store_id", "product_id", "posterior_mean", "posterior_std"]
)
global_reg_output.to_csv("clustered_reg.csv", index=False)