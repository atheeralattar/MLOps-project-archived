"""
NOTE: Scratchpad blocks are used only for experimentation and testing out code.
The code written here will not be executed as part of the pipeline.
"""
from mlflow import MlflowClient
import mlflow
import os
os.chdir('/home/src/your_codebase')
from tracking.mlflow import *
from importlib import reload
import tracking.mlflow
reload(tracking.mlflow)
experiment_name = 'Lasso'
client = MlflowClient()
mlflow.search_experiments()
client = MlflowClient()
experiment = mlflow.search_experiments()[0]
for run in client.search_runs(experiment.experiment_id):
    top_run = client.search_runs(
    experiment_ids=experiment.experiment_id,
    filter_string="metrics.r2_test > 0",
    order_by=["metrics.r2_test DESC"],
    max_results = 1    
    )
top_run