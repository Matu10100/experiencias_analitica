import os
import argparse
import wandb
import warnings
 
import numpy as np
import pandas as pd
import pickle
 
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
 
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()
 
if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"
 
def read(data_dir, filename):
    run = wandb.init()
    artifact = run.use_artifact('Housing-preprocess:latest')
    table = artifact.get(filename)
    data = table.get_dataframe()
    return data
 
def train(config, dataset):
    X_train = dataset.drop('target', axis=1)
    y_train = dataset['target']
    # Train model, get predictions
    reg = LinearRegression(**config)
    model = reg.fit(X_train, y_train)
    wandb.sklearn.plot_residuals(model, X_train, y_train)
    return model
 
def evaluate(dataset, model):
    X_test = dataset.drop('target', axis=1)
    y_test = dataset['target']
 
    y_pred = model.predict(X_test)
    r2 = metrics.r2_score(y_test, y_pred)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    return r2, MSE, y_pred
 
def train_and_log(config, experiment_id='99'):
    with wandb.init(
        project="maestria_datos", 
        name=f"Train Model ExecId-{args.IdExecution} ExperimentId-{experiment_id}", 
        job_type="train-model") as run:
        config = wandb.config
        data = run.use_artifact('Housing-preprocess:latest')
        model_config = data.metadata
        config.update(model_config)
        data_dir = data.download(root="./data/artifacts/")      
        training_set =  read(data_dir, "training.table.json")
        trained_model = train(config=config, dataset=training_set)
 
         # Save and log the trained model
        model_filename = "trained_model.pkl"
        with open(model_filename, 'wb') as model_file:
            pickle.dump(trained_model, model_file)
 
        model_artifact = wandb.Artifact(
            "trained-model", type="model",
            description="Trained LinearRegression model")
        model_artifact.add_file(model_filename)
        run.log_artifact(model_artifact)
 
    return trained_model
 
def evaluate_and_log(model, experiment_id='99'):
    with wandb.init(
        project="maestria_datos", name=f"Eval Model ExecId-{args.IdExecution} ExperimentId-{experiment_id}", 
        job_type="eval-model") as run:
 
        data = run.use_artifact('Housing-preprocess:latest')
        data_dir = data.download(root="./data/artifacts/")
        training_set =  read(data_dir, "training.table.json")
        testing_set = read(data_dir, "test.table.json")
 
        # Evaluate the model
        r2, MSE, y_pred = evaluate(testing_set, model)
        wandb.log({"Test R^2": r2, "Test MSE": MSE})
    return r2, MSE, y_pred
 
def read_raw(data_dir, filename):
    run = wandb.init()
    artifact = run.use_artifact('Housing-Raw:latest')
    table = artifact.get(filename)
    data = table.get_dataframe()
    return data
 
# Main script
parameters = ['True','False']
 
for id,parameter in enumerate(parameters):
    # Train the model and get the trained model object
    model_config = {"positive" : parameter,
             "fit_intercept": parameter,
             #"solver": 'lsqr'
             }
    trained_model = train_and_log(model_config,id)
    # Evaluate the trained model and log evaluation metrics
    evaluate_and_log(trained_model)
