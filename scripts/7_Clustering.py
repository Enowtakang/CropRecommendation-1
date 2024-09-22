import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder

"""
1. Paths
"""
data_path = "C:/Users/HP/PycharmProjects/CropRec/data/averaged_dataset.csv"
results_path = "C:/Users/HP/PycharmProjects/CropRec/results/Clustering"


"""
2. Load and Prepare dataset
"""
data = pd.read_csv(data_path)

X = data[['N', 'P', 'K', 'temperature',
          'humidity', 'ph', 'rainfall']]

y = data['label'].values


"""
3. Class counts
"""
# Count instances for each class and save to Excel


def all_class_counts():
    y2 = data['label'].value_counts()
    class_counts = y2.reset_index()
    class_counts.columns = ['Class', 'Count']

    save_path = os.path.join(
            results_path, 'Class_counts.xlsx')
    class_counts.to_excel(
        save_path, sheet_name='Counts',
        index=False, index_label=False)


"""
4. Dendrograms

Perform agglomerative clustering and 
plot dendrograms with different metrics
"""


def plot_dendrograms():

    metrics = list(
        ['euclidean', 'chebyshev', 'cityblock',
         'cosine', 'hamming',])

    for metric in metrics:
        # Perform hierarchical clustering
        linked = linkage(
            X,
            method='complete',
            metric=metric)

        # Create a new figure for each dendrogram
        plt.figure(figsize=(10, 10))

        dendrogram(
            linked,
            orientation='top',
            labels=y,
            distance_sort='descending')

        plt.title(f'Dendrogram using {metric} distance')
        plt.xlabel('Classes')
        plt.ylabel('Distance')

        # Save the dendrogram as JPG
        save_path = os.path.join(
            results_path,
            f'{str.upper(metric)}-based clustering.jpg')
        plt.savefig(save_path)
        plt.close()


# all_class_counts()
plot_dendrograms()
