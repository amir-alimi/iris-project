# iris-project
⭐ Summary of the Iris Classification Code (English)

The provided Python script is a complete machine-learning pipeline designed to compare three major classification algorithms on the Iris dataset:

Logistic Regression

Decision Tree Classifier

Support Vector Machine (SVM)

The code is written in a modular, object-oriented structure to make it clean, extendable, and professional.

🔹 1. Dataset Loading

A custom class DatasetLoader loads the Iris dataset from a CSV file located on the user’s device.
It displays the dataset shape and the first few rows to confirm successful loading.

🔹 2. Data Preprocessing

The DataPreprocessor class:

Separates features (X) and labels (y)

Normalizes all numeric features using StandardScaler
This ensures that models like SVM and Logistic Regression work correctly.

🔹 3. Model Preparation & Training

The ModelEvaluator class manages:

Initializing the three models (LogReg, Decision Tree, SVM)

Training each model on the training dataset

Storing the trained models

🔹 4. Model Evaluation

For each model:

Predictions are generated on the test set

Accuracy is calculated

A classification report (precision, recall, F1-score) is displayed

A confusion matrix heatmap is plotted for visualization

All results are stored for comparison.

🔹 5. Cross-Validation

Each model undergoes 5-fold cross-validation to check its consistency and performance stability.

🔹 6. Hyperparameter Tuning

GridSearchCV is used to tune the best hyperparameters for each model:

Logistic Regression: C parameter

Decision Tree: depth, criterion

SVM: kernel type, C

The best model configuration for each classifier is printed and returned.

🔹 7. Accuracy Comparison Chart

A bar chart visually compares the accuracy of all three models.

🔹 8. Saving the Best Model

The best-performing model (SVM in this example) is saved as a .pkl file using pickle so it can be reused later without retraining.

✔️ In summary:

This code builds a full ML classification system, including:

Data loading

Preprocessing

Model training

Evaluation

Visualization

Cross-validation

Hyperparameter optimization

Model saving

It is structured professionally and follows modern machine learning standards.