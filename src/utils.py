import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(x_train, x_test, y_train, y_test, model, param):
    try:
        report = {}
        for i in range(len(list(model))):
            model_name = list(model.keys())[i]
            model_obj = list(model.values())[i]
            param_grid = param.get(model_name, {})  # get the param grid for this model
            if param_grid:  # Only use GridSearchCV if there are params to tune
                gs = GridSearchCV(model_obj, param_grid=param_grid, cv=3)
                gs.fit(x_train, y_train)
                model_obj = gs.best_estimator_
            else:
                model_obj.fit(x_train, y_train)
            y_train_pred = model_obj.predict(x_train)
            y_test_pred = model_obj.predict(x_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)