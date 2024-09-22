import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score,
    confusion_matrix, )
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve


"""
1. Paths
"""
data_path = "C:/Users/HP/PycharmProjects/CropRec/data/crop_recommendation.csv"
results_path = "C:/Users/HP/PycharmProjects/CropRec/results/NaiveBayes"

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
# Initialize the Gaussian Naive Bayes model
model = GaussianNB()
# Fit the model on the training data
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)


"""
5. Numerical Evaluation
"""


def numerical_evals():
    # Calculate accuracy and precision
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average='weighted')
    # Save results to an Excel file
    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision'],
        'Score': [accuracy, precision]
    })
    save_path = os.path.join(
        results_path, 'NB_model_evaluation_metrics.xlsx')
    results_df.to_excel(
        save_path, index=False, index_label=False)


"""
Graphical evaluations
"""


def conf_matrix():
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Plotting Confusion Matrix
    plt.figure(figsize=(13, 13))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='coolwarm',
        xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('Gaussian Naive Bayes Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    save_path = os.path.join(
        results_path, 'NB_confusion_matrix.jpg')
    plt.savefig(save_path)  # Save as JPG file


"""
Explainability
"""


def nb_learning_curves():
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5)
    #
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    #
    plt.figure(figsize=(6, 6))
    plt.plot(train_sizes, train_scores_mean,
             label='Training Score', color='blue')
    plt.plot(train_sizes, test_scores_mean,
             label='Cross-validation Score', color='red')
    plt.title('Gaussian Naive Bayes Learning Curves')
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()

    save_path = os.path.join(
        results_path, 'NB_learning_curves.jpg')
    plt.savefig(save_path)  # Save as JPG file


# Calculate feature importance based on
# conditional probabilities for each class label.
def feature_importance(ml_model):
    features = X.columns.tolist()
    importance_dict = {}
    for i in range(len(ml_model.class_prior_)):
        importance_dict[f'Class {i}'] = ml_model.theta_[i]

    return pd.DataFrame(importance_dict, index=features)


def plot_feature_importance():
    importance_df = feature_importance(model)

    # Plotting Feature Importance Scores
    importance_df.plot(kind='bar', figsize=(12, 12))
    plt.title('Gaussian Naive Bayes Feature Importance Scores (Conditional Probabilities)')
    plt.xlabel('Features')
    plt.ylabel('Conditional Probability')
    plt.legend(title='Classes')

    save_path = os.path.join(
        results_path, 'NB_feature_importance_scores.jpg')
    plt.savefig(save_path)  # Save as JPG file


"""
Final Analysis Function
"""


def final_analysis():
    numerical_evals()
    conf_matrix()
    nb_learning_curves()
    plot_feature_importance()


final_analysis()































