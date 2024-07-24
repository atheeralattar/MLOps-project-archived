import mlflow
from sklearn.metrics import mean_squared_error, r2_score
def mlflow_default_logging(model, model_tag, X_train, y_train, X_test, y_test):
    yp_train = model.predict(X_train)
    yp_test = model.predict(X_test)
        
    # Metrics
    rmse_train = mean_squared_error(y_train, yp_train, squared=False)
    rmse_valid = mean_squared_error(y_test, yp_test, squared=False)
    mlflow.set_tag("model", model_tag)
    r2=r2_score(y_test, yp_test)
    mlflow.log_metric("rmse_train", rmse_train)
    mlflow.log_metric("rmse_valid", rmse_valid)
    mlflow.log_metric("r2_test", r2)
    #mlflow.log_figure(fig, "figure.svg")