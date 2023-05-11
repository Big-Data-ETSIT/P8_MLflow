import mlflow.pyfunc
import sys
import numpy as np

model_name ="decision_tree"
model_version = int(sys.argv[1])

laboratory_test = int(sys.argv[2])
first_test= float(sys.argv[3])
second_test= float(sys.argv[4])
days_missing = float(sys.argv[5])
hours_studied = float(sys.argv[6])
first_exam = float(sys.argv[7])

feature_columns = ["laboratory_test","first_test",
                    "second_test","days_missing","hours_studied","first_exam"]

data = np.array([laboratory_test,first_test, second_test,
                days_missing, hours_studied, first_exam]).reshape(1, -1)

tracking_uri = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(tracking_uri)

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

aprueba = "No"
if model.predict(data)[0] > 0:
    aprueba = "Sí"

print("El alumno aprueba: " + aprueba)