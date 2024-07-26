"""
This module will perform the following tasks:

"""
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tracking.mlflow import mlflow_default_logging
import os
import mlflow
from mlflow.models import infer_signature
import pickle
import json

def train(df, tracking_uri, exp_name):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)
    for model_class in [LinearRegression, Lasso, Ridge]:
        with mlflow.start_run():
                target = 'totalFare'
                X = df.drop(target, axis=1)
                mlflow.log_param('features', json.dumps(list(X.columns)))
                y = df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                train_dicts = X_train.to_dict(orient='records')
                test_dicts = X_test.to_dict(orient='records')
                dv = DictVectorizer(sparse=False)
                X_train = dv.fit_transform(train_dicts)
                X_test = dv.transform(test_dicts)
                y_train = y_train.values
                y_test = y_test.values
                
                # Fit model
                model = model_class()
                model.fit(X_train, y_train)
                mlflow.xgboost.autolog()
                signature = infer_signature(X_test, model.predict(X_test))
                mlflow.sklearn.log_model(model, "model", signature=signature)
                MODEL_TAG = model_class.__name__
                args = [model, MODEL_TAG, X_train, y_train, X_test, y_test]
                mlflow_default_logging(*args)
def train_nn(df, tracking_uri, exp_name, **kwargs):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)
    for model_class in [MLPRegressor]:
        with mlflow.start_run():
            mlflow.sklearn.autolog()
            target = 'totalFare'
            X = df.drop(target, axis=1)
            mlflow.log_param('features', json.dumps(list(X.columns)))
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            train_dicts = X_train.to_dict(orient='records')
            test_dicts = X_test.to_dict(orient='records')
            dv = DictVectorizer(sparse=False)
            X_train = dv.fit_transform(train_dicts)
            X_test = dv.transform(test_dicts)
            y_train = y_train.values
            y_test = y_test.values

            # data scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # modeling
            mlp = MLPRegressor(random_state=42, max_iter=500)
            param_grid = {'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                            'activation': ['relu', 'tanh'],
                            'alpha': [0.0001, 0.001, 0.01],
                            'learning_rate': ['constant', 'adaptive'],}   

            # grid search
            grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
            grid_search.fit(X_train_scaled, y_train)

            best_mlp = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print("Best parameters:", best_params)
            mlflow.log_param('Best parameters', json.dumps(list(best_params)))
            mlflow.log_param('data size', X_train.shape)            
            # Step 8: Make predictions using the best model
            y_pred = best_mlp.predict(X_test_scaled)

            # Step 9: Evaluate the model
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            # model = model_class()
            # signature = infer_signature(X_test, best_mlp.predict(X_test))
            # mlflow.sklearn.log_model(model, "model", signature=signature)
            # MODEL_TAG = model_class.__name__
            # args = [model, MODEL_TAG, X_train, y_train, X_test, y_test]
            # mlflow_default_logging(*args)
            print(f"Mean Squared Error: {mse}")
            print(f"R-squared Score: {r2}")
            









