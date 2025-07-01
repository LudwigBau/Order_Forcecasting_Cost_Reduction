# Import libraries:
import pandas as pd
import numpy as np
import pickle

from datetime import datetime

from src.utils.workforce_model import workforce_model
from src.utils.simulation_utils import fit_distribution

# Get today's date as a string in the format 'yymmdd' to save file accordingly
date_string = datetime.now().strftime('%y%m%d')

# import data with all models
full_back_df = pd.read_pickle("../../data/modelling_results/ens_back_results_v2.pickle")
full_pred_df = pd.read_pickle("../../data/modelling_results/ens_pred_results_v2.pickle")


# Test Case PLEASE COMMENT OUT IN ACTUAL RUN!
"""
models = ['L_6_ets_L_4_sarimax', 'actual', 'L_4_sarimax', 'L_4_Time_Momentum_Lag_lgbm', 'L_4_lstm']

full_back_df = full_back_df[models]
full_pred_df = full_pred_df[models]
"""
# Extract the 'actual' column values from both DataFrames
back_act_values = full_back_df['actual'].values
pred_act_values = full_pred_df['actual'].values

# Concatenate the two arrays into one
actuals = np.concatenate([back_act_values, pred_act_values])

# Import Forecast Results
forecast_results = pd.read_excel('../../data/modelling_results/ensemble_metrics_v2.xlsx', index_col="Unnamed: 0")

# Extract the 'actual' column values from both DataFrames
back_act_values = full_back_df['actual'].values
pred_act_values = full_pred_df['actual'].values

# Concatenate the two arrays into one
actuals = np.concatenate([back_act_values, pred_act_values])

# WORKFORCE-MODEL PARAMETERS

# Set up simulation parameters
samples = 5600  # control number of scenarios, try to take multiple of 7
verbose = True
np.random.seed(42)  # Set seed

T = 6  # Number of workdays
K = int(samples / (T + 1))  # Number of scenarios

# Cost Scenarios
c_p_l = [16, 18, 20]  # Cost per hour for planned workers (l = list)
c_e_l = [20, 22, 24]  # Cost per hour for extra workers (l = list)
c_o_l = [18, 20, 22]  # Cost per hour for overtime (l = list)

# PSI Scenarios
psi_l = [0.7, 0.8, 0.9]  # Service Level

# emax Scenarios
emax_l = [0.25, 0.5, 0.75]  # Service Level

p_p = 100  # Productivity per planned worker (filled products per hour)
p_e = 90  # Productivity per extra worker (filled products per hour)
p_o = 95  # Productivity per overtime worker (filled products per hour)

# run with all models
# define constants (middle values of list)
c_p = c_p_l[1]
c_e = c_e_l[1]
c_o = c_o_l[1]
psi = psi_l[1]

cost_i = 1

# Select Distribution for simulation
dist_name, _, goodness_of_fit_df = fit_distribution(actuals)

# Run Workforce Model on all models
full_eval_df, full_eval_dict = workforce_model(full_back_df, full_pred_df, dist_name,
                                               c_p, c_e, c_o, psi, cost_i, samples)

# SAVE SINGLE

# save Single DF
with open(f'../../data/modelling_results/workforce_results_single_all{date_string}.pickle', 'wb') as handle:
    pickle.dump(full_eval_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

# save Single DICT
with open(f'../../data/modelling_results/workforce_results_single_all_dict{date_string}.pickle', 'wb') as handle:
    pickle.dump(full_eval_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("\nSINGLE WORKFORCE EVAL FINISHED\n")
# Find Best Model

# Select best model to perform robustness test
# Define parameters for slack penalty
p_s = 90       # productivity conversion factor for slack
c_e = 22       # extra worker cost per shift
penalty = 1.25 * c_e  # penalty cost per unit slack

# Pre-calculate benchmark data (including slack penalty)
sum_bench = (np.sum(full_eval_dict["L_4_sarimax"]["cost"], axis=1) +
              (np.sum(full_eval_dict["L_4_sarimax"]["s"], axis=1) / p_s) * penalty)

#bench_data = sum_bench
bench_data = np.mean(sum_bench)

# Initialize a list to store the evaluation metrics
evaluation_list = []

# Loop through each model to evaluate
for model, variable_dict in full_eval_dict.items():
    # Compute model cost including slack penalty
    sum_data = (np.sum(variable_dict["cost"], axis=1) +
                  (np.sum(variable_dict["s"], axis=1) / p_s) * penalty)
    #model_data = sum_data
    model_data = np.mean(sum_data)
    # Calculate mean saving (in %)
    mean_saving = np.round(np.mean(((model_data - bench_data) / bench_data) * 100), 2)

    # Append evaluation metrics for this model
    evaluation_list.append({
        'model': model,
        'mean_saving': mean_saving,
    })

# Convert the list of dictionaries to a DataFrame
best_model_evaluation_df = pd.DataFrame(evaluation_list)

# Get best forecast model (actual is first, thus best forecast is second)
best_forecast_model = forecast_results.iloc[1].category

# Get the best workforce model based on lowest mean saving
best_workforce_model = best_model_evaluation_df.sort_values("mean_saving").iloc[0].model

# If the best forecast model is "actual", select the first two rows;
# otherwise, select the first row and add "actual" manually.
if best_workforce_model == "actual":
    list_of_r_models = best_model_evaluation_df.sort_values("mean_saving").iloc[:2].model.tolist()
else:
    list_of_r_models = best_model_evaluation_df.sort_values("mean_saving").iloc[:1].model.tolist()
    if "actual" not in list_of_r_models:
        list_of_r_models.append("actual")

# Add L_4_sarimax (Bench) to the list if not already present
if "L_4_sarimax" not in list_of_r_models:
    list_of_r_models.append("L_4_sarimax")

# Finally, ensure the best forecasting model is in list_of_r_models. If not, add it.
if best_forecast_model not in list_of_r_models:
    list_of_r_models.append(best_forecast_model)

# Start Robustness Test

# Select forecast models to run robust optimisation on.
robust_back_df = full_back_df[list_of_r_models]
robust_pred_df = full_pred_df[list_of_r_models]

# Initialize empty DataFrames
robust_cost_eval_df = pd.DataFrame()
robust_psi_eval_df = pd.DataFrame()
robust_emax_eval_df = pd.DataFrame()

# then run best model benchmark and perfect prediction

# Create empty dict to save scenario results
cost_scenario_dict = {}

# loop over range of scenarios
for cost_i in range(0, len(c_p_l)):
    # select scenario
    c_p_t = c_p_l[cost_i]
    c_e_t = c_e_l[cost_i]
    c_o_t = c_o_l[cost_i]

    # select psi scenario
    psi_t = psi_l[1]

    print(f"Start COST sensitivity scenario nr: {cost_i + 1}:")

    temp_robust_cost_eval_df, cost_scenario_dict[cost_i] = workforce_model(robust_back_df, robust_pred_df, dist_name,
                                                                           c_p_t, c_e_t, c_o_t, psi_t, cost_i, samples)

    # append to the robust_cost_eval_df
    robust_cost_eval_df = pd.concat([robust_cost_eval_df, temp_robust_cost_eval_df], ignore_index=True)

# Create empty dict to save scenario results
psi_scenario_dict = {}

for psi_i in range(0, len(psi_l)):
    # select scenario
    psi_t = psi_l[psi_i]

    cost_i = 1
    # keep cost constant
    c_p_t = c_p_l[1]
    c_e_t = c_e_l[1]
    c_o_t = c_o_l[1]

    print(f"Start PSI sensitivity scenario nr: {psi_i + 1}:")

    temp_robust_psi_eval_df, psi_scenario_dict[psi_i] = workforce_model(robust_back_df, robust_pred_df, dist_name,
                                                                        c_p_t, c_e_t, c_o_t, psi_t, cost_i, samples)

    # append to the robust_psi_eval_df
    robust_psi_eval_df = pd.concat([robust_psi_eval_df, temp_robust_psi_eval_df], ignore_index=True)

# concat the two robustness test dfs to get a final robustness df
robust_evaluation_df = pd.concat([robust_cost_eval_df, robust_psi_eval_df], ignore_index=True)

# Create empty dict to save scenario results
emax_scenario_dict = {}

for emax_i in range(0, len(emax_l)):
    # select scenario
    emax_t = emax_l[emax_i]

    cost_i = 1

    # keep cost constant
    c_p_t = c_p_l[1]
    c_e_t = c_e_l[1]
    c_o_t = c_o_l[1]

    # select psi scenario
    psi_t = psi_l[1]

    print(f"Start emax sensitivity scenario nr: {emax_i + 1}:")

    temp_robust_emax_eval_df, emax_scenario_dict[emax_i] = workforce_model(robust_back_df, robust_pred_df, dist_name,
                                                                           c_p_t, c_e_t, c_o_t, psi_t, cost_i, samples,
                                                                           e_max_rate=emax_t)

    # append to the robust_psi_eval_df
    robust_emax_eval_df = pd.concat([robust_emax_eval_df, temp_robust_emax_eval_df], ignore_index=True)

# concat the three robustness test dfs to get a final robustness df
robust_evaluation_df = pd.concat([robust_cost_eval_df, robust_psi_eval_df], ignore_index=True)
robust_evaluation_df = pd.concat([robust_evaluation_df, robust_emax_eval_df], ignore_index=True)

# save for today
goodness_of_fit_df.to_excel(f'../../data/modelling_results/simulation_goodness_of_fit{date_string}.xlsx',
                            index=False)



# ROBUST

# Save Summary DF
with open(f'../../data/modelling_results/workforce_results_robust{date_string}.pickle', 'wb') as handle:
    pickle.dump(robust_evaluation_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save dict
# cost
with open(f'../../data/modelling_results/workforce_results_robust_cost_dict{date_string}.pickle', 'wb') as handle:
    pickle.dump(cost_scenario_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# psi
with open(f'../../data/modelling_results/workforce_results_robust_psi_dict{date_string}.pickle', 'wb') as handle:
    pickle.dump(psi_scenario_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# e_max
with open(f'../../data/modelling_results/workforce_results_robust_emax_dict{date_string}.pickle', 'wb') as handle:
    pickle.dump(emax_scenario_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)