"""
This module will perform the following tasks:

"""
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tracking.mlflow import mlflow_default_logging, check_expirement_if_not_exist
import os
import time
import mlflow
from mlflow.models import infer_signature
import pickle
import json
import warnings


def train(df, tracking_uri):
    mlflow.autolog()
    mlflow.set_tracking_uri(tracking_uri)
    for model_class in [RandomForestRegressor, Lasso]:
        check_expirement_if_not_exist(model_class.__name__)
        with mlflow.start_run() as run:
            target = 'totalFare'
            X = df.drop(target, axis=1)
            mlflow.log_param('features', json.dumps(list(X.columns)))
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            mlflow.log_param('data size', X_train.shape)
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
            signature = infer_signature(X_test, model.predict(X_test))
            MODEL_TAG = model_class.__name__
            args = [model, MODEL_TAG, X_train, y_train, X_test, y_test]
            mlflow_default_logging(*args)
            

            
def train_nn(df, tracking_uri, **kwargs):
    warnings.filterwarnings("ignore")
    mlflow.set_tracking_uri(tracking_uri)
    for model_class in [MLPRegressor]:
        mlflow.set_experiment(model_class.__name__)
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
            param_grid = {'hidden_layer_sizes': [(100, 50)],
                            'activation': ['tanh'],
                            'alpha': [0.02],
                            'learning_rate': ['adaptive'],}   

            # grid search
            grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
            print('grid_search.fit(X_train_scaled, y_train)')
            grid_search.fit(X_train_scaled, y_train)
            best_mlp = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print("Best parameters:", best_params)
            mlflow.log_param('Best parameters', json.dumps(list(best_params)))
            mlflow.log_param('data size', X_train.shape)
            y_pred = best_mlp.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            # Step 8: Make predictions using the best model
            print(f"Test Mean Squared Error: {mse}")
            print(f"R-squared Score: {r2}")