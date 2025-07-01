# Import libraries:
import pandas as pd
import numpy as np
import pickle

from datetime import datetime

from src.utils.workforce_model import workforce_model
from src.utils.simulation_utils import fit_distribution

# import data with all models
full_back_df = pd.read_pickle("../../data/modelling_results/ens_back_results_v2.pickle")
full_pred_df = pd.read_pickle("../../data/modelling_results/ens_pred_results_v2.pickle")

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
K = int(samples/(T+1))  # Number of scenarios

# PSI Scenarios
psi_l = [0.7, 0.8, 0.9]  # Service Level

# Step 1: Define the constants (define first to loop over them)
c_p_l = [16, 18, 20]  # Cost per hour for planned workers (l = list)
c_e_l = [20, 22, 24]  # Cost per hour for extra workers (l = list)
c_o_l = [18, 20, 22]  # Cost per hour for overtime (l = list)

p_p = 100  # Productivity per planned worker (filled products per hour)
p_e = 90  # Productivity per extra worker (filled products per hour)
p_o = 95  # Productivity per overtime wor1ker (filled products per hour)

# run with all models
# define constants (middle values of list)
psi = psi_l[1]
c_p = c_p_l[1]
c_e = c_e_l[1]
c_o = c_o_l[1]

cost_i = 1

# Select Distribution for simulation
dist_name, _, goodness_of_fit_df = fit_distribution(actuals)

# Run Workforce Model on all models
full_eval_df, full_eval_dict = workforce_model(full_back_df, full_pred_df, dist_name,
                                               c_p, c_e, c_o, psi, cost_i, samples)

# Robustness Test

# Select best model to perform robustness test
# Pre-calculate benchmark data
bench_data = np.sum(full_eval_dict["L_4_sarimax"]["cost"], axis=1).flatten()

# Initialize a list to store the evaluation metrics
evaluation_list = []

# Loop through each model to evaluate
for model, variable_dict in full_eval_dict.items():
    # Extract and flatten model data
    model_data = np.sum(variable_dict["cost"], axis=1).flatten()

    # Calculate mean saving
    mean_saving = np.round(np.mean(((model_data - bench_data) / bench_data) * 100), 2)

    # Create and append a dictionary to hold evaluation metrics for this model
    evaluation_list.append({
        'model': model,
        'mean_saving': mean_saving,
    })

# Convert the list of dictionaries to a DataFrame
best_model_evaluation_df = pd.DataFrame(evaluation_list)

# Sort DataFrame and display the top two models based on mean_saving
list_of_r_models = best_model_evaluation_df.sort_values("mean_saving").iloc[:2].model.tolist()

# Add L_4_sarimax (Bench) to the list
list_of_r_models.append("L_4_sarimax")

# select forecast models to run robust optimisation on
robust_back_df = full_back_df[list_of_r_models]
robust_pred_df = full_pred_df[list_of_r_models]


# Initialize empty DataFrames
robust_cost_eval_df = pd.DataFrame()
robust_psi_eval_df = pd.DataFrame()

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

# save for today
# Get today's date as a string in the format 'yymmdd' to save file accordingly
date_string = datetime.now().strftime('%y%m%d')

goodness_of_fit_df.to_excel('../../data/modelling_results/simulation_goodness_of_fit.xlsx', index=False)

# SINGLE

# save Single DF
with open(f'../../data/modelling_results/workforce_results_single_all{date_string}.pickle', 'wb') as handle:
    pickle.dump(full_eval_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

# save Single DICT
with open(f'../../data/modelling_results/workforce_results_single_all_dict{date_string}.pickle', 'wb') as handle:
    pickle.dump(full_eval_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ROBUST

# Save DF
with open(f'../../data/modelling_results/workforce_results_robust{date_string}.pickle', 'wb') as handle:
    pickle.dump(robust_evaluation_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save dict
# cost
with open(f'../../data/modelling_results/workforce_results_robust_cost_dict{date_string}.pickle', 'wb') as handle:
    pickle.dump(cost_scenario_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# psi
with open(f'../../data/modelling_results/workforce_results_robust_psi_dict{date_string}.pickle', 'wb') as handle:
    pickle.dump(psi_scenario_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
