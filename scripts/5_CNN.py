import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
import shap
from sklearn.preprocessing import LabelEncoder


"""
1. Paths
"""
data_path = "C:/Users/HP/PycharmProjects/CropRec/data/crop_recommendation.csv"
results_path = "C:/Users/HP/PycharmProjects/CropRec/results/CNN"


"""
2. Load and Prepare dataset
"""
data = pd.read_csv(data_path)
# Split features and labels
X = data[['N', 'P', 'K', 'temperature', 'humidity',
          'ph', 'rainfall']].values
y = data['label'].values
# Encode labels using LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)


"""
3. Model building
"""
# Reshape the input data for CNN
# (assuming it is to be treated as a 2D image)
X_train_reshaped = X_train.reshape(-1, 7, 1, 1)
# Reshape to (samples, height, width, channels)
X_test_reshaped = X_test.reshape(-1, 7, 1, 1)

model = models.Sequential([
    layers.Conv2D(
        32, (3, 1), activation='relu',
        input_shape=(7, 1, 1), padding='same'),

    layers.MaxPooling2D((2, 1)),

    layers.Conv2D(
        64, (3, 1),
        activation='relu', padding='same'),

    layers.MaxPooling2D((2, 1)),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    # Output layer for multi-class classification
    layers.Dense(len(np.unique(y_encoded)),
                 activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(
    X_train_reshaped, y_train, epochs=50,
    validation_data=(X_test_reshaped, y_test))


"""
4. Model evaluation
"""
# Predictions and evaluation metrics
y_pred = np.argmax(model.predict(X_test_reshaped), axis=-1)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

"""Save results to Excel file"""

results_df = pd.DataFrame(
    {'Metric': ['Accuracy', 'Precision'],
     'Score': [accuracy, precision]})
save_path1 = os.path.join(
        results_path, 'CNN_model_evaluation_metrics.xlsx')
results_df.to_excel(
    save_path1, index=False, index_label=False)

"""Confusion Matrix Visualization"""

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(13, 13))
sns.heatmap(cm, annot=True, fmt='d', cmap='inferno',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('CNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

save_path2 = os.path.join(
        results_path, 'CNN_confusion_matrix.jpg')
plt.savefig(save_path2)
plt.close()


"""
5. Model Explainability
"""

"""Generate GradCAM"""


def get_gradcam_heatmap(model, img_array):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer("conv2d").output,
                 model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(
            np.array([img_array]))
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=0)

    gc_heatmap = tf.reduce_mean(
        tf.multiply(pooled_grads[None],
                    conv_outputs[0]), axis=-1)
    gc_heatmap = np.maximum(
        gc_heatmap.numpy(), 0) / np.max(gc_heatmap.numpy())

    return gc_heatmap


# Generate GradCAM for each class and save figures
for i in range(len(X_test)):
    heatmap = get_gradcam_heatmap(model, X_test_reshaped[i])

    plt.imshow(heatmap)
    plt.title(f'GradCAM for Class: {le.inverse_transform([y_pred[i]])[0]}')
    plt.colorbar()

    save_path3 = os.path.join(
        results_path, f'CNN_GradCAM_class_{le.inverse_transform([y_pred[i]])[0]}.jpg')
    plt.savefig(save_path3)
    plt.close()


"""Learning Curve"""

# Learning Curves Visualization
plt.plot(history.history['accuracy'],
         label='Training Accuracy')
plt.plot(history.history['val_accuracy'],
         label='Validation Accuracy')
plt.title('CNN Learning Curves')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

save_path6 = os.path.join(
        results_path, 'CNN_learning_curves.jpg')
plt.savefig(save_path6)
plt.close()


"""SHAP Analysis"""

# SHAP values calculation using DeepExplainer for CNNs
explainer = shap.DeepExplainer(model, X_train_reshaped)
shap_values = explainer.shap_values(X_test_reshaped)

# Reshape SHAP values to 2D
shap_values_reshaped = shap_values[0].reshape(
    shap_values[0].shape[0], -1)


# Save SHAP values to Excel file
shap_values_df = pd.DataFrame(
    shap_values_reshaped,
    columns=[f'Feature_{i}' for i in range(
        shap_values_reshaped.shape[1])])

save_path4 = os.path.join(
        results_path, 'CNN_shap_values.xlsx')
shap_values_df.to_excel(
    save_path4, index=False, index_label=False)

# Visualize SHAP values for each class and save figures
# Print the shape of shap_values for debugging
print([sv.shape for sv in shap_values])

for i in range(len(shap_values)):
    # Ensure there are enough SHAP values
    if shap_values[i].shape[0] > 0:
        shap.summary_plot(
            shap_values[i],
            X_test_reshaped.reshape(
                -1, X_test_reshaped.shape[-1]),
            plot_type="bar"
        )
        plt.title(f'CNN SHAP Values for Class: {le.inverse_transform([i])[0]}')
        save_path5 = os.path.join(
            results_path, f'CNN_shap_class_{le.inverse_transform([i])[0]}.jpg')
        plt.savefig(save_path5)
        plt.close()
    else:
        print(f"Skipping class {i} due to insufficient SHAP values.")


"""
Feature Importance Scores using Integrated Gradients 
(for interpretability)
"""


def integrated_gradients(ml_model, inputs):
    baseline = np.zeros(inputs.shape)
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        preds_baseline = ml_model(baseline)
        preds_input = ml_model(inputs)

    grads_input = tape.gradient(
        preds_input - preds_baseline.mean(), inputs)[0]

    return grads_input.numpy()


feature_importance_scores = integrated_gradients(
    model, X_train_reshaped)

plt.bar(range(len(
    feature_importance_scores)),
    feature_importance_scores.mean(axis=0))
plt.title('CNN Feature Importance Scores')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.xticks(range(len(
    feature_importance_scores)),
    ['N', 'P', 'K', 'Temp', 'Humidity', 'pH', 'Rainfall'])

save_path7 = os.path.join(
        results_path, 'CNN_feature_importance_scores.jpg')
plt.savefig(save_path7)
plt.close()

