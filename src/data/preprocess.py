import warnings
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
#testing
import os
import argparse
import wandb
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()
if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"
def preprocess(dataset, normalize=True):
    """
    ## Prepare the data
    """
    X = dataset.drop('target', axis=1)
    y = dataset['target']
    if normalize:
        X_normalized = (X - X.min()) / (X.max() - X.min())  # Min-Max scaling
        X = pd.DataFrame(X_normalized, columns=X.columns)  # Convert back to DataFrame
        y_normalized = (y - y.min()) / (y.max() - y.min())  # Min-Max scaling
        y = pd.DataFrame(y_normalized)  # Convert back to DataFrame

        dataset = pd.concat([X,y], axis=1)
    return dataset
def preprocess_and_log(steps):
    with wandb.init(project="maestria_datos",name=f"Preprocess Data ExecId-{args.IdExecution}", job_type="preprocess-data") as run:    
        processed_data = wandb.Artifact(
            "Housing-preprocess", type="dataset",
            description="Preprocessed California Housing dataset",
            metadata=steps)
        # ✔️ declare which artifact we'll be using
        raw_data_artifact = run.use_artifact('Housing-Raw:latest')
        # 📥 if need be, download the artifact
        raw_dataset = raw_data_artifact.download(root="./data/artifacts/")
        for split, name in zip(["training.table", "validation.table", "test.table"], df_names):
            raw_split = read(raw_dataset, split)
            processed_dataset = preprocess(raw_split, **steps)
 
            preprocessed_data = wandb.Table(data=processed_dataset.values, columns=list(processed_dataset.columns))
            # Add table to the artifact
            processed_data.add(preprocessed_data, name=f"{name}")
        run.log_artifact(processed_data)
def read(data_dir, split):
    run = wandb.init()
    artifact = run.use_artifact('Housing-Raw:latest')
    filename = split + ".json"
    table = artifact.get(filename)
    data = table.get_dataframe()
    return data
steps = {"normalize": True}
df_names = ['training', 'validation', 'test']
 
preprocess_and_log(steps)
