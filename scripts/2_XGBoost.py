import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, confusion_matrix,
    classification_report)
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import learning_curve


"""
1. Paths
"""
data_path = "C:/Users/HP/PycharmProjects/CropRec/data/crop_recommendation.csv"
results_path = "C:/Users/HP/PycharmProjects/CropRec/results/XGBoost"


"""
2. Load and Prepare dataset
"""
data = pd.read_csv(data_path)
# Features and target variable
X = data[['N', 'P', 'K', 'temperature', 'humidity',
          'ph', 'rainfall']]
y = data['label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42)


"""
3. Model building
"""
# Create the XGBoost classifier
model = XGBClassifier(
    objective='multi:softprob',
    num_class=22, max_depth=6,
    learning_rate=0.1, n_estimators=100)
# Fit the model
model.fit(X_train, y_train)


"""
4. Model evaluation
"""
# Make predictions
y_pred = model.predict(X_test)


def numerical_evaluation():
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average='weighted')
    # Save results to an Excel file
    results = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision'],
        'Score': [accuracy, precision]})

    save_path = os.path.join(
        results_path, 'XG_model_evaluation.xlsx')
    results.to_excel(save_path, index=False)


def xgboost_conf_matrix():
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(13, 13))
    sns.heatmap(conf_matrix, annot=True, fmt='d',
                cmap='magma',
                xticklabels=np.unique(y),
                yticklabels=np.unique(y))
    plt.title('XGBoost Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    save_path = os.path.join(
        results_path, 'XG_confusion_matrix.jpg')
    plt.savefig(save_path)
    plt.close()


def xgboost_learning_curves():
    (train_sizes,
     train_scores,
     test_scores) = learning_curve(
        model, X_train, y_train, cv=5)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes,
             train_mean, label='Training score')
    plt.plot(train_sizes,
             test_mean, label='Cross-validation score')
    plt.title('XGBoost Learning Curves')
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend()

    save_path = os.path.join(
        results_path, 'XG_learning_curves.jpg')
    plt.savefig(save_path)
    plt.close()


def xgboost_feature_importance_scores():
    importance = model.feature_importances_
    features = X.columns

    plt.figure(figsize=(12, 12))
    sns.barplot(x=importance, y=features, color='purple')
    plt.title('XGBoost Feature Importance Scores')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')

    save_path = os.path.join(
        results_path, 'XG_feature_importance.jpg')
    plt.savefig(save_path)
    plt.close()


numerical_evaluation()
xgboost_conf_matrix()
xgboost_learning_curves()
xgboost_feature_importance_scores()









