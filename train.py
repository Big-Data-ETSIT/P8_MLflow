import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

current_file_path = os.path.abspath(__file__)
here = os.path.dirname(current_file_path)


def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    return accuracy, precision, recall, f1


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_file = os.path.join(here, "grades.csv")
    
    data = pd.read_csv(csv_file, sep=",")
    
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    target_column = "passed_last_exam"
    feature_columns = ["laboratory_test","first_test",
                       "second_test","days_missing","hours_studied","first_exam"]

    max_depth = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    name ="decision_tree"

    tracking_uri = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(tracking_uri)


    with mlflow.start_run(run_name=name+' experiment',
                          description="DecisionTreeClassifier model (max_depth={:f}:".format(max_depth)) as run:
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=1)
        model.fit(train[feature_columns].values, train[target_column])

        test_pred = model.predict(test[feature_columns])

        (accuracy, precision, recall, f1) = eval_metrics(test[target_column], test_pred)


        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=name,
            registered_model_name=name)