import subprocess

if __name__ == "__main__":
    # Define the script's path
    classic_ts = "../models/classic_ts.py"
    tree_based_ts = "../models/tree_based_ts.py"
    lstm_ts = "../models/lstm_ts.py"
    nhits_ts = "../models/nhits_ts.py"

    # eval
    eval = "../evaluation/forecast_evaluation.py"

    # Workforce Model
    workforce = "../simulation/single_and_robust_workforce.py"

    #scripts = [classic_ts, tree_based_ts, lstm_ts, nhits_ts]
    scripts = [eval, workforce]
    #script_names = ['Classic', 'Tree Based', 'LSTM', 'Nhits']
    script_names = ["Evaluation", "Workforce"]

    for script, name in zip(scripts, script_names):

        print(f"Start: {name} Forcasting Methods")
        # Use subprocess to run the script
        subprocess.call(['python', script])