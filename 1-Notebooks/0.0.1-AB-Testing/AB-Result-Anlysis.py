import numpy as np
import pandas as pd

# Demo 17. AB Test Results Analysis.py

N_exp = 10000
N_con = 10000

# Generating Click Data
click_exp = pd.Series(np.random.binomial(1, 0.5, size=N_exp))
click_con = pd.Series(np.random.binomial(1, 0.2, size=N_con))

# Generate Group Identifier
exp_id = pd.Series(np.repeat("exp", N_exp))
con_id = pd.Series(np.repeat("con", N_con))

df_exp = pd.concat([click_exp, exp_id], axis=1)
df_con = pd.concat([click_con, con_id], axis=1)

df_exp.columns = ["click", "group"]
df_con.columns = ["click", "group"]

df_ab_test = pd.concat([df_exp, df_con], axis=0).reset_index(drop=True)
#print(df_ab_test)

