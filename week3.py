# Install required libraries
!pip install -U scikit-learn pandas matplotlib seaborn

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

# Define a function to train and evaluate logistic regression
def logistic_regression_experiment(X, y, dataset_name):
    # Step 1: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Step 2: Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 3: Initialize and train the logistic regression model
    log_reg = LogisticRegression(max_iter=10000)
    log_reg.fit(X_train, y_train)

    # Step 4: Make predictions
    y_pred = log_reg.predict(X_test)

    # Step 5: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"--- {dataset_name} Dataset ---")
    print(f"Accuracy: {accuracy:.2f}")
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Classification report
    class_report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(class_report)

    # Visualize confusion matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return accuracy

# Logistic Regression on the Iris Dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
logistic_regression_experiment(X_iris, y_iris, "Iris")

# Logistic Regression on the Wine Dataset
wine = load_wine()
X_wine = wine.data
y_wine = wine.target
logistic_regression_experiment(X_wine, y_wine, "Wine")

# Logistic Regression on the Breast Cancer Dataset
cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target
logistic_regression_experiment(X_cancer, y_cancer, "Breast Cancer")