# Import
import numpy as np
import pandas as pd
import scipy.stats

# Set-Up
samples = 2800
alpha = 3
verbose = False
np.random.seed(42)


def fit_distribution(actuals, dist_name=None):

    all_dists_df = pd.DataFrame(
        columns=['Distribution', 'MLE_Params', 'K-S_Stat', 'K-S_p-value', 'AD_Stat', 'Param1', 'Param2', 'Param3',
                 'Param_Names'])
    temp_dfs = []

    param_names_dict = {
        'norm': ['Mean', 'Std Dev'],
        'lognorm': ['Shape (s)', 'Loc', 'Scale'],
        'gamma': ['Shape (a)', 'Loc', 'Scale'],
        'weibull_min': ['Shape (c)', 'Loc', 'Scale']
    }

    if dist_name is None:
        dists = ['norm', 'lognorm', 'gamma', 'weibull_min']

        for dist_name in dists:
            dist = getattr(scipy.stats, dist_name)
            params = dist.fit(actuals)

            # Perform K-S test
            D, p_value = scipy.stats.kstest(actuals, dist_name, args=params)

            # Extract individual parameters for separate columns
            param1, param2, param3 = params if len(params) == 3 else (*params, None)

            # Get parameter names
            param_names = param_names_dict.get(dist_name, ['-'])
            param_names_str = ', '.join([f"{i + 1}:{name}" for i, name in enumerate(param_names)])

            # Add to temporary DataFrame list
            temp_df = pd.DataFrame({
                'Distribution': [dist_name],
                # 'MLE_Params': [params],
                'Param_Names': [param_names_str],
                'Param1': [param1],
                'Param2': [param2],
                'Param3': [param3],
                'K-S_Stat': [D],
                'K-S_p-value': [p_value],
            })
            temp_dfs.append(temp_df)

        # Concatenate all temporary DataFrames
        all_dists_df = pd.concat(temp_dfs, ignore_index=True)

        # Find the best distribution based on K-S test p-value
        dist_name = all_dists_df.loc[all_dists_df['K-S_p-value'].idxmax(), 'Distribution']

    # Fit the best distribution to the data
    dist = getattr(scipy.stats, dist_name)
    params = dist.fit(actuals)

    print(f"Fitting distribution: {dist_name}")
    print(f"MLE Parameters: {params}")

    return dist_name, params, all_dists_df


def simulate_actuals(actuals, dist_name, samples=2800, verbose=False):
    # Set max demand as 100% of actuals
    max_actual = np.max(actuals)  # Get the maximum value of the actuals

    _, params, _ = fit_distribution(actuals, dist_name=dist_name)
    # Generate random samples from the best-fitting distribution
    dist = getattr(scipy.stats, dist_name)

    simulated_samples = dist.rvs(size=samples, *params)
    clipped_samples = np.clip(simulated_samples, None, max_actual)
    if verbose:
        print(f"Best-fitting distribution: {dist_name}")
        print(f"MLE Parameters: {params}")

    return clipped_samples


def calc_error(real, pred):

    error = pred - real

    return error


def simulate_forecast(error, simulated_actuals, samples=samples):

    #Set min demand as 50% of actuals
    min_demand = 0

    #get stats
    normal_mean_error = np.mean(error)
    normal_std_error = np.std(error)

    # simulate error (use variance reduction of 0.5 standard-dev)
    normal_samples_error = np.random.normal(normal_mean_error, normal_std_error/2, size=samples)

    # add errors to forecast
    sim_forecast = simulated_actuals + normal_samples_error

    # Truncate the normal distribution 
    sim_forecast = np.maximum(sim_forecast, min_demand)

    return sim_forecast


def simulation_main(real, pred, dist_name, samples, verbose):
    # Set seed
    np.random.seed(42)
    error = calc_error(real, pred)
    sim_a = simulate_actuals(real, dist_name, samples=samples, verbose=verbose)
    sim_f = simulate_forecast(error, sim_a, samples=samples)

    return sim_a, sim_f


def create_weeks(back, pred):   
    # Concat backtest and predictions
    df = pd.concat([back, pred], axis=0)

    # Add a column for the weekday (to check)
    df = df.reset_index()
    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])
    # Now you can use day_name()
    df['weekday'] = df['date'].dt.day_name()

    # Set 'date' as index
    df.set_index('date', inplace=True)

    # Find the first Sunday after the first date
    start_date = df.index[0]
    while start_date.day_name() != 'Sunday':
        start_date += pd.Timedelta(days=1)

    # Find the last Saturday before the last date
    end_date = df.index[-1]
    while end_date.day_name() != 'Saturday':
        end_date -= pd.Timedelta(days=1)

    # Slice the dataframe between the start and end dates
    week_df = df.loc[start_date: end_date]

    # print(week_df)
    # Drop Weekday column
    week_df = week_df.drop(columns="weekday")
    # Create a list to hold the weekly data
    weekly_data = []

    # Iterate over the weeks
    for week in np.array_split(week_df, len(week_df) // 7):
        weekly_data.append(week.values)

    return weekly_data

