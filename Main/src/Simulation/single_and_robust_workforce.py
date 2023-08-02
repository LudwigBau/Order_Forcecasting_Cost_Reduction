# Import libraries:
import pandas as pd
import numpy as np
import pickle

from datetime import datetime

from src.utils.workforce_model import workforce_model

# import data with all models
full_back_df = pd.read_pickle("../../data/modelling_results/ens_back_results_v2.pickle")
full_pred_df = pd.read_pickle("../../data/modelling_results/ens_pred_results_v2.pickle")

# select forecast models to run robust optimisation on
robust_back_df = full_back_df[["actual", "L_4_sarimax", "L_4_Time_Momentum_Lag_lgbm"]]
robust_pred_df = full_pred_df[["actual", "L_4_sarimax", "L_4_Time_Momentum_Lag_lgbm"]]

# WORKFORCE-MODEL PARAMETERS

# Set up simulation parameters
samples = 2800  # control number of scenarios, try to take multiple of 7
alpha = 3  # Control tail of simulated forecast
verbose = True  # Print gamma parameters
np.random.seed(42)  # Set seed

T = 6  # Number of workdays
K = int(samples/(T+1))  # Number of scenarios

# PSI Scenarios
psi_l = [0.7, 0.8, 0.9]  # Service Level

# Step 1: Define the constants (define first to loop over them)
c_p_l = [16, 18, 20]  # Cost per hour for planned workers (l = list)
c_e_l = [20, 22, 24]  # Cost per hour for extra workers (l = list)
c_o_l = [18, 20, 22]  # Cost per hour for overtime (l = list)

p_p = 12  # Productivity per planned worker (filled products per hour)
p_e = 10  # Productivity per extra worker (filled products per hour)
p_o = 11  # Productivity per overtime worker (filled products per hour)

L = 10  # Maximum shift length

# run with all models
# define constants (middle values of list)
psi = psi_l[1]
c_p = c_p_l[1]
c_e = c_e_l[1]
c_o = c_o_l[1]

cost_i = 1

full_eval_df = workforce_model(full_back_df, full_pred_df, c_p, c_e, c_o, psi, cost_i)

# save for today
# Get today's date as a string in the format 'yymmdd' to save file accordingly
date_string = datetime.now().strftime('%y%m%d')

# Combine it with your base filename
filename = f'../../data/modelling_results/workforce_results_single_all{date_string}.pickle'

## Save your model
with open(filename, 'wb') as handle:
    pickle.dump(full_eval_df, handle, protocol=pickle.HIGHEST_PROTOCOL)


# then run best model benchmark and perfect prediction

# Initialize empty DataFrames
robust_cost_eval_df = pd.DataFrame()
robust_psi_eval_df = pd.DataFrame()

# then run best model benchmark and perfect prediction

# loop over range of scenarios
for cost_i in range(0, len(c_p_l)):
    # select scenario
    c_p = c_p_l[cost_i]
    c_e = c_e_l[cost_i]
    c_o = c_o_l[cost_i]

    # select psi scenario
    psi = psi_l[1]

    print(f"Start COST sensitivity scenario nr: {cost_i + 1}:")

    temp_robust_cost_eval_df = workforce_model(robust_back_df, robust_pred_df, c_p, c_e, c_o, psi, cost_i)

    # append to the robust_cost_eval_df
    robust_cost_eval_df = pd.concat([robust_cost_eval_df, temp_robust_cost_eval_df], ignore_index=True)

for psi_i in range(0, len(psi_l)):
    # select scenario
    psi = psi_l[psi_i]

    # keep cost constant
    cost_i = 1
    c_p = c_p_l[cost_i]
    c_e = c_e_l[cost_i]
    c_o = c_o_l[cost_i]

    print(f"Start PSI sensitivity scenario nr: {psi_i + 1}:")

    temp_robust_psi_eval_df = workforce_model(robust_back_df, robust_pred_df, c_p, c_e, c_o, psi, cost_i)

    # append to the robust_psi_eval_df
    robust_psi_eval_df = pd.concat([robust_psi_eval_df, temp_robust_psi_eval_df], ignore_index=True)

# concat the two robustness test dfs to get a final robustness df

robust_evaluation_df = pd.concat([robust_cost_eval_df, robust_psi_eval_df], ignore_index=True)

print(robust_evaluation_df)

# save for today
# Get today's date as a string in the format 'yymmdd' to save file accordingly
date_string = datetime.now().strftime('%y%m%d')

# Combine it with your base filename
filename = f'../../data/modelling_results/workforce_results_robust{date_string}.pickle'

## Save your model
with open(filename, 'wb') as handle:
    pickle.dump(robust_evaluation_df, handle, protocol=pickle.HIGHEST_PROTOCOL)