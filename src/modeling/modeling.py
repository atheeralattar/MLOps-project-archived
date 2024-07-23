"""
This module will perform the following tasks:

"""
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
import os
import mlflow
import pickle

def model(df, tracking_uri, exp_name ):
    """
    Modeling the data using several models (LR, RIDGE, LASSO)    
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)
    
    directory = 'models'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # data prep
    target = 'totalFare'
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    train_dicts = X_train.to_dict(orient='records')
    test_dicts = X_test.to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    X_test = dv.fit_transform(test_dicts)
    y_train = y_train.values
    y_test = y_test.values

    # with mlflow.start_run():
    #     mlflow.set_tag("developer", "atheer")
    #     lr = LinearRegression()
    #     lr.fit(X_train, y_train)
    #     y_pred = lr.predict(X_test)
    #     accuracy['LR'] = root_mean_squared_error(y_test, y_pred)
    #     with open('models/lin_reg.bin', 'wb') as f_out:
    #         pickle.dump((dv, lr), f_out)
    #     mlflow.log_metric("rmse", accuracy['LR'])
    #     mlflow.log_artifact(local_path="models/lin_reg.bin", artifact_path="models_pickle")
    
    mlflow.sklearn.autolog()

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    rr = Ridge()
    rr.fit(X_train, y_train)
    y_pred_rr = rr.predict(X_test)

    lasso = Lasso()
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)

