# ----------------------------------------------------------
#   Classification Model Comparison on Local Iris Dataset
# ----------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# 1) بارگذاری دیتاست از مسیر شخصی
# ----------------------------------------------------------

data_path = "/Users/amirmohammad/Documents/kiyana solgi 23/iris.csv"

df = pd.read_csv(data_path)

print("نمونه دیتا:")
print(df.head())

# ستون‌های دیتای استاندارد Iris:
# sepal_length, sepal_width, petal_length, petal_width, species

X = df.iloc[:, 0:4]    # ویژگی‌ها
y = df.iloc[:, 4]      # کلاس

# ----------------------------------------------------------
# 2) Train/Test
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------------
# 3) نرمال‌سازی
# ----------------------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------
# 4) مدل‌ها
# ----------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=300),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM (RBF Kernel)": SVC(kernel="rbf")
}

results = {}

# ----------------------------------------------------------
# 5) آموزش + تست + Confusion Matrix
# ----------------------------------------------------------

for name, model in models.items():

    if name == "Decision Tree":
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print("\n==============================")
    print("Model:", name)
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# ----------------------------------------------------------
# 6) مقایسه نهایی دقت مدل‌ها
# ----------------------------------------------------------

print("\n==============================")
print("   Final Accuracy Comparison")
print("==============================")

for name, acc in results.items():
    print(f"{name}: {acc}")
