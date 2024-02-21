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
 
        dataset = pd.concat([X,y], axis=1)
    return dataset
 
def preprocess_and_log(steps):
 
    with wandb.init(
        project="maestria_datos",
        name=f"Preprocess Data ExecId-{args.IdExecution}", job_type="preprocess-data") as run:    

        # 🏺 create our Artifact 
        processed_data = wandb.Artifact(
            "Housing-preprocess", type="dataset",
            description="Preprocessed California Housing dataset",
            metadata=steps)
        # ✔️ declare which artifact we'll be using
        raw_data_artifact = run.use_artifact('Housing-Raw:latest')
 
        # 📥 if need be, download the artifact
        raw_dataset = raw_data_artifact.download(root="./data/artifacts/")

       # Create the directory for processed data if it doesn't exist
        processed_data = "processed_data"
        if not os.path.exists(processed_data):
            os.makedirs(processed_data)
         
        for split in ["training.table", "validation.table", "test.table"]:
            raw_split = read(raw_dataset, split)
            processed_dataset = preprocess(raw_split, **steps)
 
            # Convert DataFrame to JSON and save
            processed_data_json_path = os.path.join("processed_data", f"{split}.json")
            processed_dataset.to_json(processed_data_json_path, orient="records")
            # Log JSON file as an artifact
            processed_data.add_file(processed_data_json_path, name=f"{split}.json")
 
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
#columns = ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude','target'] 
preprocess_and_log(steps)
