# Import
import numpy as np
import pandas as pd
from scipy.stats import gamma

# Steps
samples = 2800
alpha = 3
verbose = False
np.random.seed(42)  # You can choose any number for the seed

def simulate_actuals(actuals, alpha=alpha, samples=samples, verbose=verbose):
    # Estimate min and max (with slack)
    min_demand = 0.8 * actuals.min()
    max_demand = 1.2 * actuals.max()

    # Estimate parameters for the gamma distribution
    gamma_shape_est, _, gamma_scale_est = gamma.fit(actuals, floc=0)  # We ignore the original loc

    # Adjust the shape parameter to make the gamma distribution more extreme
    alpha = alpha  # Modify this value to make the distribution more extreme

    # Generate random samples from the gamma distribution with the adjusted shape parameter
    gamma_samples_est = gamma.rvs(a=alpha, loc=min_demand, scale=gamma_scale_est, size=samples)  # Set loc to min_demand

    # Truncate the gamma samples to the desired range
    truncated_gamma_samples_est = np.clip(gamma_samples_est, min_demand, max_demand)

    # Normal estimates
    normal_mean_est = np.mean(actuals)
    normal_std_est = np.std(actuals)

    # Print Parameters
    if verbose == True:
        print("Estimated Parameters:")
        print("Normal Distribution - Mean:", normal_mean_est, "Standard Deviation:", normal_std_est)
        print("Gamma Distribution - Shape:", gamma_shape_est, "Scale:", gamma_scale_est)
        print("Alpha:", alpha)

    # Return Value
    return truncated_gamma_samples_est


def calc_error(real, pred):

    error = pred - real

    return error


def simulate_forecast(error, simulated_actuals, samples=samples):
    #Set min demand as 50% of actuals
    min_demand = 0.5 * simulated_actuals.min()

    #get stats
    normal_mean_error = np.mean(error)
    normal_std_error = np.std(error)

    # sample
    normal_samples_error = np.random.normal(normal_mean_error, normal_std_error, size=samples)

    # add errors to forecast
    sim_forecast = simulated_actuals + normal_samples_error

    # Make sure sim_forecast does not go below 0.5*min_demand
    sim_forecast = np.maximum(sim_forecast, min_demand)

    return sim_forecast


def simulation_main(real, pred, alpha=alpha, samples=samples, verbose=verbose):
    # Set seed
    np.random.seed(42)
    error = calc_error(real, pred)
    sim_a = simulate_actuals(real, alpha=alpha, samples=samples, verbose=verbose)
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

