"""
This is a boilerplate pipeline 'somthing'
generated using Kedro 0.18.11
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
from mlflow.models import infer_signature, set_signature
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def process_data(data):
    data.index = data["datetime"]
    data = data.drop("datetime", axis=1)
    return data


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def do_rf(data, params):
    x = data.drop("Volume", axis=1)
    y = x["Close"].diff()
    y = y.iloc[1:]
    x = x.iloc[1:]
    y = np.where(y >= 0, 1, 0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)

    predict = rf.predict(x_test)

    (rmse, mae, r2) = eval_metrics(y_test, predict)
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # signature = infer_signature(x_train, rf.predict(x_train))
    # run = mlflow.last_active_run()
    # model_uri = f"runs:/{run.info.run_id}/model"
    # set_signature(model_uri=model_uri, signature=signature)

    metric = {
        "RMSE" : {"value" : rmse, "step" : 0},
        "MAE" : {"value" : mae, "step" : 0},
        "R2" : {"value" : r2, "step" : 0}
    }

    return {"clf" : rf,
            "model_metrics" : metric,
            }


def do_signature(clf, data):
    x = data.drop("Volume", axis=1)
    y = x["Close"].diff()
    y = y.iloc[1:]
    x = x.iloc[1:]
    y = np.where(y >= 0, 1, 0)

    signature = infer_signature(x, clf.predict(x))
    run = mlflow.last_active_run()
    model_uri = f"runs:/{run.info.run_id}/model"
    set_signature(model_uri=model_uri, signature=signature)
    return True


def check(clf, data):
    x = data.drop("Volume", axis=1)
    y = x["Close"].diff()
    y = y.iloc[1:]
    x = x.iloc[1:]
    y = np.where(y >= 0, 1, 0)

    aa = pd.DataFrame([clf.score(x, y)])

    return aa
