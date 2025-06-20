import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(x_train, x_test, y_train, y_test, model):
    try:
        report = {}  # <-- define report dictionary
        for i in range(len(list(model))):
            model_obj = list(model.values())[i]
            model_obj.fit(x_train, y_train)
            y_train_pred = model_obj.predict(x_train)
            y_test_pred = model_obj.predict(x_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(model.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)