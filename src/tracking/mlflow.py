import mlflow
from mlflow import MlflowClient
from sklearn.metrics import mean_squared_error, r2_score
def mlflow_default_logging(model, model_tag, X_train, y_train, X_test, y_test):
    yp_train = model.predict(X_train)
    yp_test = model.predict(X_test)
        
    # Metrics
    rmse_train = mean_squared_error(y_train, yp_train, squared=False)
    rmse_valid = mean_squared_error(y_test, yp_test, squared=False)
    mlflow.set_tag("model", model_tag)
    r2=r2_score(y_test, yp_test)
    mlflow.log_metric("r2_test", r2)
    
def check_registry_if_not_exist(experiment):
    """ checks and creates a registry if not created
        and register the top run from that experiment
    Args:
        experiment_name (obj): experiment object
    """
    client = MlflowClient()
    registered_models = client.search_registered_models()
    top_run = pick_top_run(experiment, 'r2_test')
    model_uri = f"runs:/{top_run.info.run_id}/model"
    
    # If the experiment does not exist, create it
    if registered_models is None:
       registered_models = mlflow.register_model(model_uri, experiment.expermint_name)
    
        


def check_expirement_if_not_exist(experiment_name):
    """ checks if an experiment is there before creating one

    Args:
        experiment_name (str): experiment name to check
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    # If the experiment does not exist, create it
    if experiment is None:
       experiment = mlflow.create_experiment(experiment_name)
       mlflow.set_experiment(experiment_name)
       print(f'created experiment: {experiment_name}')
    else:
        experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_name)
        print('experiment exists already.')





def pick_top_run(experiment: object, parameter):
    """ Picks the top run for a given experiment

    Args:
        experiment (obj): experiment object coming from search expirements
        parameter (str): parameter needed to sort the experiments according to

    Returns:
        top_run (obj): run object from search run
    """
    client = MlflowClient()
    for run in client.search_runs(experiment.experiment_id):
        top_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        filter_string="metrics.r2_test > 0",
        order_by=["metrics.r2_test DESC"],
        max_results = 1    
        )
    return top_run
    
    
    
    
def model_registry():
    client = MlflowClient()
    experiments = mlflow.search_experiments()
    for experiment in experiments:
        pick_top_run(experiment, 'r2_test')
        
    
    
    # Query all runs in the experiment and output the top 3
    top_runs = client.search_runs(
    experiment_ids=experiment_id,
    order_by=["metrics.r2_test DESC"],
    filter_string="metrics.r2_test > .4",
    max_results = 3    
    )
    
    # register these 3 models if they are not registered.
    model_registry_name = "mlflow_test"
    for run in top_runs:
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        model_version = None
        mlflow.register_model(model_uri, "mlflow-testing")


        # # Check if the model is already registered
        # if run_id not in registered_models:
        #     # Register the model
        #     result = mlflow.register_model(model_uri, model_registry_name)
        #     model_version = result.version
        #     new_models.append((run_id, model_version))
        #     print(f"Registered new model '{model_registry_name}' with version {model_version}.")
        # else:
        #     model_version = registered_models[run_id]
        
        # print(f"Run ID: {run_id}, Registered Version: {model_version}")
        