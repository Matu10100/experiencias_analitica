import warnings
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Testing
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

def load(train_size=.8):
    """
    # Load the data
    """
    housing = datasets.fetch_california_housing()
    
    # Convertir los datos y las características en un DataFrame de pandas
    df = pd.DataFrame(housing.data, columns=housing.feature_names)

    # Agregar la columna 'target' al DataFrame para los valores objetivo
    df['target'] = housing.target

    # Define the proportion of data you want in each subset
    train_ratio = 0.8  
    test_ratio = 0.2 

    # Calculate the sizes of each subset
    num_total_samples = len(df)
    num_train = int(train_ratio * num_total_samples)
    num_test = int(test_ratio * num_total_samples)

    # Split the data into subsets
    train_data = df[:num_train]
    test_data = df[num_train:]

    X = train_data.drop('target', axis=1)
    y = train_data['target']
    X, y = X[::2], y[::2]  # subsample for faster demo
    wandb.errors.term._show_warnings = False
    # ignore warnings about charts being built from subset of data

    # the data, split between train and test sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    training_set = pd.concat([X_train,y_train], axis=1)
    validation_set = pd.concat([X_val,y_val], axis=1)
    test_set = test_data

    datasets = [training_set, validation_set, test_set]
    return datasets

def load_and_log():
    # 🚀 start a run, with a type to label it and a project it can call home
    with wandb.init(
        project="maestria_datos",
        name=f"Load Raw Data ExecId-{args.IdExecution}", job_type="load-data") as run:
        
        datasets = load()  # separate code for loading the datasets
        names = ["training", "validation", "test"]

        # 🏺 create our Artifact
        raw_data = wandb.Artifact(
            "Housing-Raw", type="dataset",
            description="raw housing dataset, split into train/val/test",
            metadata={"source": "sklearn.datasets.fetch_california_housing",
                      "sizes": [len(dataset) for dataset in datasets]})

        # ✍️ Save the artifact to W&B.
        run.log_artifact(raw_data)

# testing
load_and_log()
