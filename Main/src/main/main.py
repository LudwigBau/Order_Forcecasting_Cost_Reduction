# Description: Python script that imports and runs the main functions from each of the scripts in the desired order.
# Desired order: Data -> Model -> Evaluation -> Simulation

# library imports
import pandas as pd
import numpy as np

# All data imports
from data.clean_data import clean_data
from data.pre_process_data import pre_process_data
from data.split_data import split_data
from data.feature_selection import feature_selection

# all model imports
from model.train_model import train_model

# all evaluation imports
from evaluation.evaluate_model import evaluate_model

# all simulation imports
from simulation.run_simulation import run_simulation

def main():
    # Data processing
    clean_data()
    pre_process_data()
    split_data()
    feature_selection()

    # Model training
    train_model()

    # Evaluation
    evaluate_model()

    # Simulation
    run_simulation()

if __name__ == "__main__":
    main()