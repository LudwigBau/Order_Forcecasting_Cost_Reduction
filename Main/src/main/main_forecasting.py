import subprocess
import pandas as pd

if __name__ == "__main__":
    # Define the script's path
    classic_ts = "../models/classic_ts.py"
    tree_based_ts = "../models/tree_based_ts.py"
    lstm_ts = "../models/lstm_ts.py"
    nhits_ts = "../models/nhits_ts.py"

    # eval
    eval = "../evaluation/forecast_evaluation.py"

    # Workforce Model
    workforce = "../simulation/workforce.py"

    # select correct src
    
    #scripts = [classic_ts, tree_based_ts, lstm_ts, nhits_ts, workforce]
    scripts = [workforce]
    #script_names = ['Classic', 'Tree Based', 'LSTM', 'Nhits', 'workforce']
    script_names = ["Workforce"]

    for script, name in zip(scripts, script_names):

        print(f"Start: {name} Forcasting Methods")
        # Use subprocess to run the script
        subprocess.call(['python', script])
