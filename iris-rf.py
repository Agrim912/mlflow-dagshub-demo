import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='Agrim912', repo_name='mlflow-dagshub-demo', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/Agrim912/mlflow-dagshub-demo.mlflow')

# Load the Iris dataset
iris = load_iris()

X=iris.data
y=iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_estimators = 30
max_depth = 15

# apply mlflow

# experiment name
mlflow.set_experiment("iris-dt")


# with mlflow.start_run(experiment_id="610707189609199498"):
# with mlflow.start_run(run_name="<give-custom-name>"):
with mlflow.start_run():
    rf=RandomForestClassifier(n_estimators=n_estimators,emax_depth=max_depth, random_state=42)

    rf.fit(X_train, y_train)

    y_pred=rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

    cm=confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    mlflow.log_artifact(__file__)

    mlflow.sklearn.log_model(rf, "random forest model")

    mlflow.set_tag("author", "rahul")
    mlflow.set_tag("model", "RandomForestClassifier")

    print("accuracy:", accuracy)