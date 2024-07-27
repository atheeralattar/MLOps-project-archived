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
    
def model_registry(exp_name):
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(exp_name)
    experiment_id = experiment.experiment_id
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(exp_name)
    else:
        experiment_id = experiment.experiment_id

    # Query all runs in the experiment
    
    runs = client.search_runs(
    experiment_ids=experiment_id,
    
    max_results=3,)
    for run in runs:
        print(run.data.metrics['r2_test'])
    # Sort the runs by R2 test score and select the top three
    # Assuming the metric is named 'r2_test'
    top_runs = runs#.sort_values("metrics.r2_test", ascending=False).head(3)
    # # Define the names for the top 3 models
    top_model_names = ["Top_Model_1", "Top_Model_2", "Top_Model_3"]
    
    # for index, (_, run) in enumerate(top_runs.iterrows()):
    #     print(run.data.metrics)[1]['r2_test']
    #     #print(f"run id: {run.info.run_id}, rmse: {run.data.metrics['r2_test']:.4f}")
    # #     #r2_score = run.data.metrics["r2_test"]
    # #     model_name = top_model_names[index]
        
    # #     # Load the model from the run
    #     model_uri = f"runs:/{run_id}/model"
        
    # #     # Check if the model already exists
    #     try:
    #         existing_versions = client.get_latest_versions(model_name)
    #     except mlflow.exceptions.RestException:
    #         existing_versions = []
        
    #     if existing_versions:
    #         # If the model exists, create a new version
    #         model_version = mlflow.register_model(model_uri, model_name)
    #         print(f"Created new version of existing model: {model_name}, version: {model_version.version}, R2 score: {r2_score}")
    #     else:
    #         # If the model doesn't exist, register it
    #         model_version = mlflow.register_model(model_uri, model_name)
    #         print(f"Registered new model: {model_name}, version: {model_version.version}, R2 score: {r2_score}")
            
    #     # Update the model description
    #     client.update_registered_model(
    #         name=model_name,
    #         description=f"Model with R2 test score: {r2_score}"
    #     )
        
    #     # Add tags to the model version
    #     client.set_model_version_tag(
    #         name=model_name,
    #         version=model_version.version,
    #         key="r2_test_score",
    #         value=str(r2_score)
    #     )
        
    #     # Transition the model to 'Production' stage
    #     client.transition_model_version_stage(
    #         name=model_name,
    #         version=model_version.version,
    #         stage="Production"
    #     )
        
    #     # Archive older versions
    #     for old_version in existing_versions:
    #         if old_version.version != model_version.version:
    #             client.transition_model_version_stage(
    #                 name=model_name,
    #                 version=old_version.version,
    #                 stage="Archived"
    #             )

    # print("Top 3 models have been updated and registered.")




    