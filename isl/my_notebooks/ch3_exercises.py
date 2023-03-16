"""_summary_
"""

# %%

from pandas import read_csv
from statsmodels.api import OLS, add_constant

# %%

#%%
DATA_PATH = r"D:\Documents\Codes\Windows\an_introduction_to_statistical_learning\isl\data\raw\Boston.csv"
boston_data = read_csv(DATA_PATH, index_col=[0])
# %%


#%%

boston_data.describe().T

# %%
X = boston_data[["lstat"]]
X = add_constant(X)
y = boston_data[["medv"]]

# %%

lm = OLS(endog=y, exog=X)
result = lm.fit()
# %%
result.params
# %%
print(result.t_test([1, 0]))
# %%
print(result.summary())

# %%
boston_data[["lstat"]]
# %%
result.predict()

# %%
