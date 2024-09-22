import os.path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score, recall_score, )
from sklearn.model_selection import learning_curve


"""
1. Paths
"""
data_path = "C:/Users/HP/PycharmProjects/CropRec/data/crop_recommendation.csv"
results_path = "C:/Users/HP/PycharmProjects/CropRec/results/RandomForest"

"""
2. Load data
"""
data = pd.read_csv(data_path)

"""
3. Train-test splitting
"""
X = data[['N', 'P', 'K', 'temperature',
          'humidity', 'ph', 'rainfall']]
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

"""
4. Model initialization and training
"""
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

"""
5. Model evaluation
"""
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)

accuracy = rf_model.score(X_test, y_test)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall, }

# Save metrics to Excel file


def save_metrics():
    metrics_df = pd.DataFrame(
        metrics.items(), columns=['Metric', 'Value'])

    save_path = os.path.join(
        results_path, '1_RF_model_metrics.xlsx')

    metrics_df.to_excel(
        save_path,
        index=False, index_label=False)


"""
6. Model evaluation Plots
"""


def confusion_matrix_plot():
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(13, 13))
    sns.heatmap(
        conf_matrix, annot=True, fmt='d', cmap='Blues',
        xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('Random Forest Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    save_path = os.path.join(
        results_path, 'RF_confusion_matrix.jpg')

    plt.savefig(save_path)
    plt.close()


"""
Model explainability plots
"""


def learning_curves_plot():
    train_sizes, train_scores, test_scores = learning_curve(
        rf_model, X_train, y_train, cv=5)

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.plot(train_sizes, train_scores_mean,
             label='Training Score')
    plt.plot(train_sizes, test_scores_mean,
             label='Cross-validation Score')
    plt.title('Random Forest Learning Curves')
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend()

    save_path = os.path.join(
        results_path, '_RF_learning_curve.jpg')

    plt.savefig(save_path)
    plt.close()


def feat_importance_scores():
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Plotting feature importance scores
    plt.figure(figsize=(10, 10))
    sns.barplot(x=importances[indices], y=X.columns[indices])
    plt.title('Random Forest Feature Importance Scores')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')

    save_path = os.path.join(
        results_path, '_RF_feature_importance_scores.jpg')

    plt.savefig(save_path)
    plt.close()


"""
Final Analysis Function
"""


def final_analysis():
    save_metrics()
    confusion_matrix_plot()
    learning_curves_plot()

    feat_importance_scores()


final_analysis()






