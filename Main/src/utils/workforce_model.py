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

# 1. Simulate (2800 scenarios per week per method) based on data
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

import gurobipy as gp
from gurobipy import GRB

from src.utils.simulation_utils import simulation_main
from src.utils.simulation_utils import create_weeks

# MODEL SETUP

# Set up simulation parameters
samples = 2800  # control number of scenarios, try to take multiple of 7
alpha = 3  # Control tail of simulated forecast
verbose = True  # Print gamma parameters
np.random.seed(42)  # Set seed

T = 6  # Number of workdays
K = int(samples/(T+1))  # Number of scenarios (T+1 = one week)

p_p = 12  # Productivity per planned worker (filled products per hour)
p_e = 10  # Productivity per extra worker (filled products per hour)
p_o = 11  # Productivity per overtime worker (filled products per hour)

cost_i = 1  # take the middle of range (0,2)

L = 8

def workforce_model(full_back_df, full_pred_df, c_p, c_e, c_o, psi, cost_i):

    back_act_df = full_back_df["actual"]
    pred_act_df = full_pred_df["actual"]

    actual_array = create_weeks(back_act_df, pred_act_df)

    # Initialize an empty DataFrame to store the evaluation metrics
    evaluation_df = pd.DataFrame()

    # Set up simulation data
    for model_index, model in enumerate(full_back_df.columns, start=1):

        back_df = full_back_df[model].copy()
        pred_df = full_pred_df[model].copy()

        # Join back and pred, cut into weeks and add sunday and monday
        weeks_array = create_weeks(back_df, pred_df)

        for week_index, (pred_week, actual_week) in enumerate(zip(weeks_array, actual_array), start=1):

            print(f"Starting simulation for week {week_index} out of {len(weeks_array)}, "
                  f"model {model_index} out of {len(full_back_df.columns)}: {model}")

            sim_a, sim_f = simulation_main(actual_week, pred_week, samples=samples, verbose=True)

            a_t = sim_a.reshape(T+1, K)
            d_t = sim_f.reshape(T+1, K)

            # Add the values from Sunday to Monday for each week
            a_t[1] += a_t[0]
            d_t[1] += d_t[0]

            # Drop the jobs for Sunday
            a_t = np.delete(a_t, 0, axis=0)
            d_t = np.delete(d_t, 0, axis=0)

            # Set Up Workforce Model:
            # Create empty list that stores cost values:
            all_cost = []

            # Step2: Create the model
            m = gp.Model("WorkforceModel")
            m.setParam('Threads', 1)
            m.ModelSense = GRB.MINIMIZE

            # Step3: Define decision variables:

            # First stage decision variables: Number of planned workers
            w_p = m.addMVar((T, K), vtype=GRB.CONTINUOUS, name="w_p")

            # Second stage decision variables: Number of extra workers and overtime (h)
            w_e = m.addMVar((T, K), vtype=GRB.CONTINUOUS, name="w_e")
            y_o = m.addMVar((T, K), vtype=GRB.CONTINUOUS, name="y_o")
            z = m.addMVar((T, K), vtype=GRB.CONTINUOUS, name="z")
            v = m.addMVar((T, K), vtype=GRB.CONTINUOUS, name="v")

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
            # It is possible to schedule 10% more workers than expected
            # This is represented by the first stage constraints:

            # Constraint: Number of planned workers should be enough to meet the expected demand
            for t in range(T):
                for k in range(K):
                    m.addConstr(w_p[t, k] * p_p * L == d_t[t, k], name=f"wp_constraint_up_t{t}_k{k}")

            # Step 6: Define Second Stage Constraints

            # in the second stage, you adjust the plan based on the actual demand (a_t).
            # This is represented by the second stage constraints:

            # Constraint: A given percentage (psi) of the total demand (daily demand and backlogs) are
            # filled by extra workers in addition to planned workers’ overtime

            for t in range(T):  # Start from 0 because there can be a backlog on the first day
                for k in range(K):
                    if t > 0:
                        m.addConstr(z[t, k] - v[t, k] == a_t[t, k] + z[t - 1, k] -
                                    (w_p[t, k] * p_p * L + y_o[t, k] * p_o + w_e[t, k] * p_e * L),
                                    name=f"backlog_constraint_t{t}_k{k}")
                    else:
                        m.addConstr(z[t, k] - v[t, k] == a_t[t, k] -
                                    (w_p[t, k] * p_p * L + y_o[t, k] * p_o + w_e[t, k] * p_e * L),
                                    name=f"backlog_constraint_t{t}_k{k}")

            m.addConstrs((z[t, k] >= 0 for t in range(T) for k in range(K)))  # z is non-negative
            m.addConstrs((v[t, k] >= 0 for t in range(T) for k in range(K)))  # v is non-negative


            # Constraint: The total productivity should be at least psi percent of the total demand
            for t in range(T):
                for k in range(K):
                    m.addConstr(w_p[t, k] * p_p * L + y_o[t, k] * p_o + w_e[t, k] * p_e * L >=
                                psi * a_t[t, k] + z[t-1, k],
                                name=f"service_level_constraint_t{t}_k{k}")

            # Constrain: the total number of extra workers is 0 or positive
            for t in range(T):
                for k in range(K):
                    m.addConstr(w_e[t, k] >= 0, name=f"extra_worker_constraint{t}_k{k}")

            # Constraint: Amount of overtime is limited to 20% of the total shift length of a worker
            for t in range(T):
                for k in range(K):
                    m.addConstr(y_o[t, k] <= 0.2 * w_p[t, k] * L, name=f"overtime_limit_constraint_t{t}_k{k}")

            for t in range(T):
                for k in range(K):
                    m.addConstr(y_o[t, k] >= 0, name=f"overtime_limit_constraint_t{t}_k{k}")

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

            for t in range(T):
                for k in range(K): all_cost.append(c_p * w_p[t, k].x * L + c_e * w_e[t, k].x * L +
                                                   c_o * y_o[t, k].x)

            # Results

            # Calculate and print the average cost
            average_cost = m.ObjVal / (T * K)
            print(f'Average cost: {average_cost}')

            std_cost = np.std(all_cost)
            print(f'Std cost: {std_cost}')

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
                'cost_scenario_values': f'c_p_{c_p}_c_e_{c_e}_c_o_{c_o}',
                'avg_cost': average_cost,
                'std_cost': std_cost,
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

    return evaluation_df



