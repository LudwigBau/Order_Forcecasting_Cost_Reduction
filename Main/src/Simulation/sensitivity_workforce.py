# Description:

# Write down steps to take to model the workforce model from "Enhancing fulfilment operation using ml"
# There are several sections to be addressed:

# Simulating Scenarios:
# Create artificial actual values based on distribution in a given week
# Create artificial forecasts based on error distribution in a given week

# First Stage Optimisation:
# Schedule planned workforce based on expected demand (forecast)
# Calculate demanded work in hours
# Fit number of workers on hours (per worker 10h)

# Second Stage Optimisation:
# Three options:
# 1. no change, planned workforce can tackle the demand
# 2. overtime, planned workforce use
# 3. extra workers, if overtime is not sufficient to fill todays demand, it exceeds tomorrow's overtime capacity
#                   then add extra workforce for tomorrow's shift.
# Test for different service levels. fill rate of 85, 90, 95 %

# Problem Description:

# I have six days of forecast.
# I plan my workforce based on the forecast
# Each day I get to know the actual demand and can adapt my workforce by overtime or extra workers on the next day to
# fill a service level of X %


# Workflow:
# Use functional programming paradigm:

# 1. Simulate (2000 scenarios per week per method) based on data
# 2. Multistep multistage optimisation
# 3. Save Results (nr. planned workers, nr. extra workers, total cost)
# 4. Analyse distribution of the three variables
# 5. Compare average cost per method.


# How To Optimise using Gurobi:

# Define decision variable
# Define constants
# Define First Stage Objective Function
# Define Constraints of first obj. function
# Define Second Stage Objective Function
# Define Constraints of second obj. function
# Save and analyse results

# Import libraries:
import pandas as pd
import numpy as np
import pickle

from datetime import datetime

import gurobipy as gp
from gurobipy import GRB

from src.utils.simulation_utils import simulation_main
from src.utils.simulation_utils import create_weeks


# import data
full_back_df = pd.read_pickle("../../data/modelling_results/ens_back_results_v2.pickle")
full_pred_df = pd.read_pickle("../../data/modelling_results/ens_pred_results_v2.pickle")

# select first columns
full_back_df = full_back_df[["actual", "L_4_sarimax", "L_4_Time_Momentum_Lag_lgbm"]]
full_pred_df = full_pred_df[["actual", "L_4_sarimax", "L_4_Time_Momentum_Lag_lgbm"]]


back_act_df = full_back_df["actual"]
pred_act_df = full_pred_df["actual"]

actual_array = create_weeks(back_act_df, pred_act_df)

# Initialize an empty DataFrame to store the evaluation metrics
evaluation_df = pd.DataFrame()

# Get today's date as a string in the format 'yymmdd' to save file accordingly
date_string = datetime.now().strftime('%y%m%d')

# MODEL SETUP

# Set up simulation parameters
samples = 2800  # control number of scenarios, try to take multiple of 7
alpha = 3  # Control tail of simulated forecast
verbose = True  # Print gamma parameters
np.random.seed(42)  # Set seed

T = 6  # Number of workdays
K = int(samples/(T+1))  # Number of scenarios

psi_l = [0.7, 0.8, 0.9]  # Service Level

# Step 1: Define the constants (define first to loop over them)
c_p_l = [16, 18, 20]  # Cost per hour for planned workers (l = list)
c_e_l = [20, 22, 24]  # Cost per hour for extra workers (l = list)
c_o_l = [18, 20, 22]  # Cost per hour for overtime (l = list)

p_p = 12  # Productivity per planned worker (filled products per hour)
p_e = 10  # Productivity per extra worker (filled products per hour)
p_o = 11  # Productivity per overtime worker (filled products per hour)

L = 10  # Maximum shift length

for cost_i in range(0, len(c_p_l)):

    # select scenario
    c_p = c_p_l[cost_i]
    c_e = c_e_l[cost_i]
    c_o = c_o_l[cost_i]

    # select psi scenario
    psi = psi_l[1]

    print(f"Start COST sensitivity scenario nr: {cost_i+1}:")

    # Set up simulation data
    for model_index, model in enumerate(full_back_df.columns, start=1):

        back_df = full_back_df[model].copy()
        pred_df = full_pred_df[model].copy()

        # Join back and pred, cut into weeks and add sunday and monday
        weeks_array = create_weeks(back_df, pred_df)

        for week_index, (pred_week, actual_week) in enumerate(zip(weeks_array, actual_array), start=1):

            print(f"Starting simulation for week {week_index} out of {len(weeks_array)}, "
                  f"model {model_index} out of {len(full_back_df.columns)}: {model}")

            sim_a, sim_f = simulation_main(actual_week, pred_week, samples=samples, verbose=False)

            a_t = sim_a.reshape(T+1, K)
            d_t = sim_f.reshape(T+1, K)

            # Add the values from Sunday to Monday for each week
            a_t[1] += a_t[0]
            d_t[1] += d_t[0]

            # Drop the jobs for Sunday
            a_t = np.delete(a_t, 0, axis=0)
            d_t = np.delete(d_t, 0, axis=0)

            # Set Up Workforce Model:

            # Step2: Create the model

            m = gp.Model("WorkforceModel")
            m.ModelSense = GRB.MINIMIZE

            # Step3: Define decision variables:

            # First stage decision variables: Number of planned workers
            w_p = m.addMVar((T, K), vtype=GRB.CONTINUOUS, name="w_p")

            # Second stage decision variables: Number of extra workers and overtime (h)
            w_e = m.addMVar((T, K), vtype=GRB.CONTINUOUS, name="w_e")
            y_o = m.addMVar((T, K), vtype=GRB.CONTINUOUS, name="y_o")

            # Step 3: Define First Stage Objective
            # objective function of the overall problem and min labour cost for expected demand for each workday

            # Objective functions

            # Sum over all time Periods
            first_stage_cost = gp.quicksum(w_p[t, k] * c_p * L for t in range(T) for k in range(K))
            # Sum over all time Periods
            overtime_cost = gp.quicksum(y_o[t, k] * c_o for t in range(T) for k in range(K))
            # Sum over all time Periods
            extra_worker_cost = gp.quicksum(w_e[t, k] * c_e * L for t in range(T) for k in range(K))

            # Objective function: Minimize the sum of the costs of planned workers, overtime, and extra workers
            m.setObjective(first_stage_cost + overtime_cost + extra_worker_cost, GRB.MINIMIZE)

            # Step 4: Define First Stage Constraints

            # In the code, the planned workers (w_p) are determined in the first stage the forecasted demand (d_t).
            # This is represented by the first stage constraints:

            # Constraint: Number of planned workers should be enough to meet the expected demand
            for t in range(T):
                for k in range(K):
                    m.addConstr(w_p[t, k] * p_p * L <= 1.1 * d_t[t, k], name=f"wp_constraint_up_t{t}_k{k}")

            # Constraint: Number of planned workers should be at least 0.9 of expected demand
            for t in range(T):
                for k in range(K):
                    m.addConstr(w_p[t, k] * p_p * L >= 1 * d_t[t, k], name=f"wp_constraint_down_t{t}_k{k}")

            # Constraint: Shift length of any worker does not exceed 10 hours
            for t in range(T):
                for k in range(K):
                    m.addConstr(L <= 10, name=f"shift_length_constraint_t{t}_k{k}")

            # Step 6: Define Second Stage Constraints

            # in the second stage, you adjust the plan based on the actual demand (a_t).
            # This is represented by the second stage constraints:

            # Constraint: A given percentage (psi) of the total demand (daily demand and backlogs) are
            # filled by extra workers in addition to planned workers’ overtime

            # Decision variable: Backlog
            z = m.addMVar((T, K), vtype=GRB.CONTINUOUS, name="z")
            # Decision variable: Overcapacity
            v = m.addMVar((T, K), vtype=GRB.CONTINUOUS, name="v")  # Auxiliary variable

            for t in range(T):  # Start from 0 because there can be a backlog on the first day
                for k in range(K):
                    if t > 0:
                        m.addConstr(z[t, k] - v[t, k] == a_t[t, k] + z[t - 1, k] - (
                                    w_p[t, k] * p_p * L + y_o[t, k] * p_o + w_e[t, k] * p_e * L),
                                    name=f"backlog_constraint_t{t}_k{k}")
                    else:
                        m.addConstr(
                            z[t, k] - v[t, k] == a_t[t, k] - (w_p[t, k] * p_p * L + y_o[t, k] * p_o + w_e[t, k] * p_e * L),
                            name=f"backlog_constraint_t{t}_k{k}")

            m.addConstrs((z[t, k] >= 0 for t in range(T) for k in range(K)))  # z is non-negative
            m.addConstrs((v[t, k] >= 0 for t in range(T) for k in range(K)))  # v is non-negative


            # Constraint: The total productivity should be at least psi percent of the total demand
            for t in range(T):  # We can now use T because we have w_e[t] for the last day
                for k in range(K):
                    m.addConstr(w_p[t, k] * p_p * L + y_o[t, k] * p_o + w_e[t, k] * p_e * L >= psi * (a_t[t, k] + z[t-1, k]),
                                name=f"service_level_constraint_t{t}_k{k}")

            for t in range(T):
                for k in range(K):
                    m.addConstr(w_e[t, k] >= 0, name=f"extra_worker_constraint{t}_k{k}")

            # Constraint: Amount of overtime is limited to 20% of the total shift length of a worker
            for t in range(T):
                for k in range(K):
                    m.addConstr(y_o[t, k] <= 0.1 * w_p[t, k] * L, name=f"overtime_limit_constraint_t{t}_k{k}")

            # Constraint: No backlog on the last day
            m.addConstr(z[T-1, :] == 0, name="no_backlog_last_day")

            # Optimize the model
            m.optimize()

            print(GRB.OPTIMAL)
            # Check if the model was solved to optimality
            if m.status == GRB.OPTIMAL:
                print('Optimal solution found')
            else:
                print("No optimal solution found.")
                m.computeIIS()
                m.write("model.ilp")

            # Results

            # Calculate and print the average cost
            average_cost = m.ObjVal / (T * K)
            print(f'Average cost: {average_cost}')

            # Calculate and print the average number of planned workers
            average_planned_workers = np.sum(w_p.X) / (T * K)
            print(f'Average number of planned workers: {average_planned_workers}')

            # Calculate and print the average number of extra workers
            average_extra_workers = np.sum(w_e.X) / (T * K)
            print(f'Average number of extra workers: {average_extra_workers}')

            # Calculate and print the average number of overtime hours
            average_overtime = np.sum(y_o.X) / (T * K)
            print(f'Average number of overtime hours: {average_overtime}')

            # Calculate and print the average number of backlog
            average_backlog= np.sum(z.X) / (T * K)
            print(f'Average number of backlog: {average_backlog}')

            # Calculate and print the average number of overcapacity
            average_overcap = np.sum(v.X) / (T * K)
            print(f'Average number of overcap: {average_overcap}')

            # evaluation
            data_dict = {
                'model': model,
                'week': week_index,
                'psi_scenario': psi,
                'cost_scenario': cost_i,
                'cost_scenario_values': f'c_p_{c_p}_c_e_{c_e}_c_o_{c_o}',  # added this line
                'avg_cost': average_cost,
                'avg_planned_workers': average_planned_workers,
                'avg_extra_workers': average_extra_workers,
                'avg_overtime': average_overtime,
                'avg_backlog': average_backlog,
                'avg_overcap': average_overcap
            }

            # create a dataframe with your data
            temp_df = pd.DataFrame(data_dict, index=[0])

            # concatenate the new dataframe to the existing one
            evaluation_df = pd.concat([evaluation_df, temp_df], ignore_index=True)


# PSI SCENARIO

for psi_i in range(0, len(psi_l)):

    # select scenario
    psi = psi_l[psi_i]

    # keep cost constant
    c_p = c_p_l[1]
    c_e = c_e_l[1]
    c_o = c_o_l[1]


    print(f"Start PSI sensitivity scenario nr: {psi_i + 1}:")

    # Set up simulation data
    for model_index, model in enumerate(full_back_df.columns, start=1):

        back_df = full_back_df[model].copy()
        pred_df = full_pred_df[model].copy()

        # Join back and pred, cut into weeks and add sunday and monday
        weeks_array = create_weeks(back_df, pred_df)

        for week_index, (pred_week, actual_week) in enumerate(zip(weeks_array, actual_array), start=1):

            print(f"Starting simulation for week {week_index} out of {len(weeks_array)}, "
                  f"model {model_index} out of {len(full_back_df.columns)}: {model}")

            sim_a, sim_f = simulation_main(actual_week, pred_week, samples=samples, verbose=False)

            a_t = sim_a.reshape(T + 1, K)
            d_t = sim_f.reshape(T + 1, K)

            # Add the values from Sunday to Monday for each week
            a_t[1] += a_t[0]
            d_t[1] += d_t[0]

            # Drop the jobs for Sunday
            a_t = np.delete(a_t, 0, axis=0)
            d_t = np.delete(d_t, 0, axis=0)

            # Set Up Workforce Model:

            # Step2: Create the model

            m = gp.Model("WorkforceModel")
            m.ModelSense = GRB.MINIMIZE

            # Step3: Define decision variables:

            # First stage decision variables: Number of planned workers
            w_p = m.addMVar((T, K), vtype=GRB.CONTINUOUS, name="w_p")

            # Second stage decision variables: Number of extra workers and overtime (h)
            w_e = m.addMVar((T, K), vtype=GRB.CONTINUOUS, name="w_e")
            y_o = m.addMVar((T, K), vtype=GRB.CONTINUOUS, name="y_o")

            # Step 3: Define First Stage Objective
            # objective function of the overall problem and min labour cost for expected demand for each workday

            # Objective functions

            # Sum over all time Periods
            first_stage_cost = gp.quicksum(w_p[t, k] * c_p * L for t in range(T) for k in range(K))
            # Sum over all time Periods
            overtime_cost = gp.quicksum(y_o[t, k] * c_o for t in range(T) for k in range(K))
            # Sum over all time Periods
            extra_worker_cost = gp.quicksum(w_e[t, k] * c_e * L for t in range(T) for k in range(K))

            # Objective function: Minimize the sum of the costs of planned workers, overtime, and extra workers
            m.setObjective(first_stage_cost + overtime_cost + extra_worker_cost, GRB.MINIMIZE)

            # Step 4: Define First Stage Constraints

            # In the code, the planned workers (w_p) are determined in the first stage the forecasted demand (d_t).
            # This is represented by the first stage constraints:

            # Constraint: Number of planned workers should be enough to meet the expected demand
            for t in range(T):
                for k in range(K):
                    m.addConstr(w_p[t, k] * p_p * L <= 1.1 * d_t[t, k], name=f"wp_constraint_up_t{t}_k{k}")

            # Constraint: Number of planned workers should be at least 0.9 of expected demand
            for t in range(T):
                for k in range(K):
                    m.addConstr(w_p[t, k] * p_p * L >= 1 * d_t[t, k], name=f"wp_constraint_down_t{t}_k{k}")

            # Constraint: Shift length of any worker does not exceed 10 hours
            for t in range(T):
                for k in range(K):
                    m.addConstr(L <= 10, name=f"shift_length_constraint_t{t}_k{k}")

            # Step 6: Define Second Stage Constraints

            # in the second stage, you adjust the plan based on the actual demand (a_t).
            # This is represented by the second stage constraints:

            # Constraint: A given percentage (psi) of the total demand (daily demand and backlogs) are
            # filled by extra workers in addition to planned workers’ overtime

            # Decision variable: Backlog
            z = m.addMVar((T, K), vtype=GRB.CONTINUOUS, name="z")
            # Decision variable: Overcapacity
            v = m.addMVar((T, K), vtype=GRB.CONTINUOUS, name="v")  # Auxiliary variable

            for t in range(T):  # Start from 0 because there can be a backlog on the first day
                for k in range(K):
                    if t > 0:
                        m.addConstr(z[t, k] - v[t, k] == a_t[t, k] + z[t - 1, k] - (
                                w_p[t, k] * p_p * L + y_o[t, k] * p_o + w_e[t, k] * p_e * L),
                                    name=f"backlog_constraint_t{t}_k{k}")
                    else:
                        m.addConstr(
                            z[t, k] - v[t, k] == a_t[t, k] - (
                                        w_p[t, k] * p_p * L + y_o[t, k] * p_o + w_e[t, k] * p_e * L),
                            name=f"backlog_constraint_t{t}_k{k}")

            m.addConstrs((z[t, k] >= 0 for t in range(T) for k in range(K)))  # z is non-negative
            m.addConstrs((v[t, k] >= 0 for t in range(T) for k in range(K)))  # v is non-negative

            # Constraint: The total productivity should be at least psi percent of the total demand
            for t in range(T):  # We can now use T because we have w_e[t] for the last day
                for k in range(K):
                    m.addConstr(w_p[t, k] * p_p * L + y_o[t, k] * p_o + w_e[t, k] * p_e * L >= psi * (
                                a_t[t, k] + z[t - 1, k]),
                                name=f"service_level_constraint_t{t}_k{k}")

            for t in range(T):
                for k in range(K):
                    m.addConstr(w_e[t, k] >= 0, name=f"extra_worker_constraint{t}_k{k}")

            # Constraint: Amount of overtime is limited to 20% of the total shift length of a worker
            for t in range(T):
                for k in range(K):
                    m.addConstr(y_o[t, k] <= 0.1 * w_p[t, k] * L, name=f"overtime_limit_constraint_t{t}_k{k}")

            # Constraint: No backlog on the last day
            m.addConstr(z[T - 1, :] == 0, name="no_backlog_last_day")

            # Optimize the model
            m.optimize()

            print(GRB.OPTIMAL)
            # Check if the model was solved to optimality
            if m.status == GRB.OPTIMAL:
                print('Optimal solution found')
            else:
                print("No optimal solution found.")
                m.computeIIS()
                m.write("model.ilp")

            # Results

            # Calculate and print the average cost
            average_cost = m.ObjVal / (T * K)
            print(f'Average cost: {average_cost}')

            # Calculate and print the average number of planned workers
            average_planned_workers = np.sum(w_p.X) / (T * K)
            print(f'Average number of planned workers: {average_planned_workers}')

            # Calculate and print the average number of extra workers
            average_extra_workers = np.sum(w_e.X) / (T * K)
            print(f'Average number of extra workers: {average_extra_workers}')

            # Calculate and print the average number of overtime hours
            average_overtime = np.sum(y_o.X) / (T * K)
            print(f'Average number of overtime hours: {average_overtime}')

            # Calculate and print the average number of backlog
            average_backlog = np.sum(z.X) / (T * K)
            print(f'Average number of backlog: {average_backlog}')

            # Calculate and print the average number of overcapacity
            average_overcap = np.sum(v.X) / (T * K)
            print(f'Average number of overcap: {average_overcap}')

            # evaluation
            data_dict = {
                'model': model,
                'week': week_index,
                'psi_scenario': psi,
                'cost_scenario': 1,
                'cost_scenario_values': f'c_p_{c_p}_c_e_{c_e}_c_o_{c_o}',  # added this line
                'avg_cost': average_cost,
                'avg_planned_workers': average_planned_workers,
                'avg_extra_workers': average_extra_workers,
                'avg_overtime': average_overtime,
                'avg_backlog': average_backlog,
                'avg_overcap': average_overcap
            }

            # create a dataframe with your data
            temp_df = pd.DataFrame(data_dict, index=[0])

            # concatenate the new dataframe to the existing one
            evaluation_df = pd.concat([evaluation_df, temp_df], ignore_index=True)

print(evaluation_df)

# Combine it with your base filename
filename = f'../../data/modelling_results/workforce_results_robust{date_string}.pickle'

## Save your model
with open(filename, 'wb') as handle:
    pickle.dump(evaluation_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
