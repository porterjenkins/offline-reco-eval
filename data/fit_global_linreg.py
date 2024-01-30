import pandas as pd
from sklearn.linear_model import LinearRegression

fpath = "./msd_2024-01-30.csv"
min_sample_tol = 10

msd = pd.read_csv(fpath)
msd = msd[~pd.isnull(msd['num_facings'])]
print(msd.columns)

global_reg_output = []

for prod, dta in msd[['product_id', 'payoff', 'num_facings']].groupby("product_id"):
    if dta.shape[0] < min_sample_tol:
        continue
    reg = LinearRegression()
    X = dta['num_facings'].values.reshape(-1,1)
    y = dta['payoff'].values.reshape(-1,1)

    reg.fit(X, y)
    print(prod, reg.coef_)
    global_reg_output.append(
        [prod,reg.coef_[0][0], 0.0]
    )


global_reg_output = pd.DataFrame(
    global_reg_output,
    columns=["product_id", "posterior_mean", "posterior_std"]
)
global_reg_output.to_csv("global_reg.csv")