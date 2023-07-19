## Create steps for to dos

import numpy as np
import gurobipy as gp
from gurobipy import GRB

if __name__ == "__main__":

    # Step 1: Define the constants

    T = 6  # Number of workdays

    c_p = 12  # Cost per hour for planned workers
    c_e = 18  # Cost per hour for extra workers
    c_o = 16  # Cost per hour for overtime

    p_p = 12  # Productivity per planned worker (filled products per hour)
    p_e = 10  # Productivity per extra worker (filled products per hour)
    p_0 = 11  # Productivity per overtime worker (filled products per hour)

    L = 10  # Maximum shift length

    K = 50  # Number of scenarios

    # Step X: Define Scenarios

    d_xi = np.random.normal(19000, 600, (T, K))  # Predicted demand

    d_eta = np.random.normal(20000, 800, (T, K))  # Actual Demand

    p = np.array([1.0 / K] * K)  # Probability of a scenario happening (normal distribution)

    # Step 2: Create the model
    m = gp.Model("WorkforcePlanning")
    m.ModelSense = GRB.MINIMIZE
    m.setParam('OutputFlag', 0)  # Telling gurobi to not be verbose
    m.params.logtoconsole = 0

    # Step 3: Define the variables

    # First stage decision variables: Number of planned workers
    w_p = m.addMVar((T,), vtype=GRB.CONTINUOUS, name="w_p")

    # Second stage decision variables: Number of extra workers and overtime
    w_e = m.addMVar((T, K), vtype=GRB.CONTINUOUS, name="w_e")
    y = m.addMVar((T, K), vtype=GRB.CONTINUOUS, name="y")

    # Step 4: Define the objective function
    m.setObjective(c_p * w_p.sum() * L + gp.quicksum(
        c_e * w_e[t, k] * L + c_o * y[t, k] * p_0 * p[k] for t in range(T) for k in range(K)))

    # Step 5: Add the constraints
    # Shift length constraints
    m.addConstrs((w_p[t] * L <= d_xi[t, k] for t in range(T) for k in range(K)))
    m.addConstrs((w_e[t, k] * L <= d_eta[t, k] for t in range(T) for k in range(K)))

    # Demand constraints
    m.addConstrs(
        (w_p[t] * L * p_p + w_e[t, k] * L * p_e + y[t, k] * p_0 >= d_eta[t, k] for t in range(T) for k in range(K)))

    # Overtime constraints
    m.addConstrs((y[t, k] <= 0.2 * w_p[t] * L for t in range(T) for k in range(K)))

    ## Last day constraints
    m.addConstr(w_e[-1, :].sum() == 0)

    # Step 6: Solve the model
    m.optimize()

    # Step 7: Print the results
    if m.status == GRB.OPTIMAL:
        print(f"Total cost: ${np.round(m.ObjVal, 2)}")
        for t in range(T):
            print(f"Day {t + 1}:")
            print(f"  Planned workers: {w_p[t].x}")
            for k in range(K):
                print(f"  Scenario {k + 1}:")
                print(f"    Extra workers: {w_e[t, k].x}")
                print(f"    Overtime: {y[t, k].x}")
    else:
        print("No optimal solution found.")

    # Print the optimal values of the decision variables
    m.printAttr('X')

    # Calculate and print the average cost
    average_cost = m.objVal / (T * K)
    print(f'Average cost: {average_cost}')

    # Calculate and print the average number of planned workers
    average_planned_workers = np.sum(w_p.X) / (T * K)
    print(f'Average number of planned workers: {average_planned_workers}')

    # Calculate and print the average number of extra workers
    average_extra_workers = np.sum(w_e.X) / (T * K)
    print(f'Average number of extra workers: {average_extra_workers}')

    # Calculate and print the average number of overtime hours
    average_overtime = np.sum(y.X) / (T * K)
    print(f'Average number of overtime hours: {average_overtime}')