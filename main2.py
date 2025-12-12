# =============================================================
#   Iris Classification - Professional ML Pipeline
#   Logistic Regression + Decision Tree + SVM
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# =============================================================
#   Dataset Loader
# =============================================================

class DatasetLoader:

    def __init__(self, path):
        self.path = path

    def load(self):
        try:
            df = pd.read_csv(self.path)
            print("\n===== Dataset Loaded Successfully =====")
            print(df.head(), "\n")
            print("Shape:", df.shape)
            return df
        except Exception as e:
            print("❌ Error Loading Dataset:", e)
            return None


# =============================================================
#   Preprocessor
# =============================================================

class DataPreprocessor:

    def __init__(self, df):
        self.df = df

    def process(self):
        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, y


# =============================================================
#   Model Evaluator (LogReg, DT, SVM)
# =============================================================

class ModelEvaluator:

    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=300),
            "Decision Tree": DecisionTreeClassifier(),
            "SVM": SVC()
        }
        self.results = {}

    def train(self, X_train, y_train):
        trained_models = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            trained_models[name] = model
        return trained_models

    def evaluate(self, trained_models, X_test, y_test):
        for name, model in trained_models.items():
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            self.results[name] = acc

            print("\n============================")
            print(f"📌 Model: {name}")
            print("Accuracy:", acc)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, cmap="Blues")
            plt.title(f"Confusion Matrix - {name}")
            plt.show()

        return self.results

    def cross_validate(self, X, y):
        print("\n=========== Cross Validation ===========")
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=5)
            print(f"{name} → Mean: {scores.mean():.3f}, Std: {scores.std():.3f}")

    def tune(self, X_train, y_train):
        print("\n=========== Hyperparameter Tuning ===========")

        params = {
            "Logistic Regression": {
                "C": [0.1, 1, 5, 10]
            },
            "Decision Tree": {
                "max_depth": [2, 3, 5, 7, 10],
                "criterion": ["gini", "entropy"]
            },
            "SVM": {
                "C": [0.1, 1, 5],
                "kernel": ["linear", "rbf"]
            }
        }

        best_models = {}

        for name, model in self.models.items():
            print(f"\n🔍 Tuning {name} ...")
            grid = GridSearchCV(model, params[name], cv=5)
            grid.fit(X_train, y_train)
            print("Best Parameters:", grid.best_params_)
            best_models[name] = grid.best_estimator_

        return best_models


# =============================================================
#   Plot Accuracy Comparisons
# =============================================================

def plot_accuracy(results):
    names = list(results.keys())
    accs = list(results.values())

    plt.figure(figsize=(10, 5))
    plt.bar(names, accs)
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.grid(axis='y')
    plt.show()


# =============================================================
#   Save Model
# =============================================================

def save_model(model, name="best_model.pkl"):
    with open(name, "wb") as f:
        pickle.dump(model, f)
    print(f"\n💾 Saved as {name}")


# =============================================================
#   Main Program
# =============================================================

def main():

    path = "/Users/amirmohammad/Documents/kiyana solgi 23/iris.csv"

    loader = DatasetLoader(path)
    df = loader.load()
    if df is None:
        return

    processor = DataPreprocessor(df)
    X, y = processor.process()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    evaluator = ModelEvaluator()

    print("\n===== Training =====")
    trained = evaluator.train(X_train, y_train)

    print("\n===== Evaluation =====")
    results = evaluator.evaluate(trained, X_test, y_test)

    print("\n===== Accuracy Comparison =====")
    plot_accuracy(results)

    print("\n===== Cross Validation =====")
    evaluator.cross_validate(X, y)

    print("\n===== Hyperparameter Tuning =====")
    best_models = evaluator.tune(X_train, y_train)

    save_model(best_models["SVM"], "best_svm_model.pkl")


# Run
main()
