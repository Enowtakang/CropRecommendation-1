import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, confusion_matrix)


"""
1. Paths
"""
data_path = "C:/Users/HP/PycharmProjects/CropRec/data/crop_recommendation.csv"
results_path = "C:/Users/HP/PycharmProjects/CropRec/results/MLP"


"""
2. Load and Prepare dataset
"""
data = pd.read_csv(data_path)
# Features and labels
X = data[['N', 'P', 'K', 'temperature', 'humidity',
          'ph', 'rainfall']]
y = data['label']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


"""
3. Model building
"""
# Build the MLP classifier
mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)


"""
4. Model evaluation
"""
# Predictions
y_pred = mlp.predict(X_test_scaled)


def numerical_evaluations():
    # Model evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average='weighted')
    # Save results to an Excel file
    metrics_df = pd.DataFrame(
        {'Metric': ['Accuracy', 'Precision'],
         'Score': [accuracy, precision]})

    save_path = os.path.join(
        results_path, 'MLP_model_evaluation_metrics.xlsx')

    metrics_df.to_excel(save_path,
                        index=False, index_label=False)


def mlp_confusion_matrix():
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(13, 13))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='viridis',
        xticklabels=np.unique(y),
        yticklabels=np.unique(y))
    plt.title('MLP Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    save_path = os.path.join(
        results_path, 'MLP_confusion_matrix.jpg')
    plt.savefig(save_path)
    plt.close()


def learning_curves_mlp():
    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    train_scores = list()
    test_scores = list()

    for size in train_sizes:
        (X_train_subset, _,
         y_train_subset, _) = train_test_split(
            X_train_scaled, y_train,
            train_size=size)

        mlp.fit(X_train_subset, y_train_subset)
        train_scores.append(accuracy_score(
            y_train_subset,
            mlp.predict(X_train_subset)))
        test_scores.append(accuracy_score(
            y_test, mlp.predict(X_test_scaled)))

    plt.figure(figsize=(6, 6))
    plt.plot(
        train_sizes, train_scores,
        label='Training Accuracy', marker='o')
    plt.plot(train_sizes, test_scores,
             label='Testing Accuracy', marker='o')
    plt.title('MLP Learning Curves')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    save_path = os.path.join(
        results_path, 'MLP_learning_curves.jpg')
    plt.savefig(save_path)
    plt.close()


def feature_importance_scores():
    # Feature Importance Scores
    # using Permutation Importance
    result = permutation_importance(
        mlp, X_test_scaled, y_test,
        n_repeats=30, random_state=42)

    # Check the result
    print(result.importances_mean)

    # Plot feature importance scores
    importance_df = pd.DataFrame(
        {'Feature': X.columns,
         'Importance': result.importances_mean})

    # Check the DataFrame
    print(importance_df)

    importance_df.sort_values(
        by='Importance', ascending=False).plot(
        kind='bar', x='Feature',
        legend=False,
        title='MLP Feature Importance Scores',
        figsize=(12, 12))
    plt.ylabel('Importance Score')

    save_path = os.path.join(
        results_path, 'MLP_feature_importance.jpg')
    plt.savefig(save_path)
    plt.close()


numerical_evaluations()
mlp_confusion_matrix()
learning_curves_mlp()
feature_importance_scores()








































