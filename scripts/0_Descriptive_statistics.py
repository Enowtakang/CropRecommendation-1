import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


"""
1. Paths
"""
data_path = "C:/Users/HP/PycharmProjects/CropRec/data/crop_recommendation.csv"
results_path = "C:/Users/HP/PycharmProjects/CropRec/results/DescriptiveStatistics"


"""
2. Load and Prepare dataset
"""
data = pd.read_csv(data_path)


"""
Work proper
"""
# Get the unique classes
classes = data['label'].unique()


def make_plot_and_stats():
    # Initialize a list to store the statistics for each class
    stats_list = list()

    # Loop through each class
    for cls in classes:
        # Isolate rows belonging to the current class
        class_df = data[data['label'] == cls]

        # Generate the pairplot
        pairplot = sns.pairplot(class_df, diag_kind='kde')
        pairplot.fig.suptitle(
            f'Pairplot for Class {str.upper(cls)}', y=1.02)

        save_path1 = os.path.join(
            results_path,
            f'Pairplot_for_class_{str.upper(cls)}.png')
        pairplot.savefig(save_path1)
        plt.close(pairplot.fig)

        # Compute statistics for each numerical feature
        stats = {
            'Class': cls,
            'N_max': class_df['N'].max(),
            'N_min': class_df['N'].min(),
            'N_range': class_df['N'].max() - class_df['N'].min(),
            'N_mean': class_df['N'].mean(),
            'N_variance': class_df['N'].var(),
            'N_std_dev': class_df['N'].std(),
            'P_max': class_df['P'].max(),
            'P_min': class_df['P'].min(),
            'P_range': class_df['P'].max() - class_df['P'].min(),
            'P_mean': class_df['P'].mean(),
            'P_variance': class_df['P'].var(),
            'P_std_dev': class_df['P'].std(),
            'K_max': class_df['K'].max(),
            'K_min': class_df['K'].min(),
            'K_range': class_df['K'].max() - class_df['K'].min(),
            'K_mean': class_df['K'].mean(),
            'K_variance': class_df['K'].var(),
            'K_std_dev': class_df['K'].std(),
            'temperature_max': class_df['temperature'].max(),
            'temperature_min': class_df['temperature'].min(),
            'temperature_range': class_df['temperature'].max() - class_df['temperature'].min(),
            'temperature_mean': class_df['temperature'].mean(),
            'temperature_variance': class_df['temperature'].var(),
            'temperature_std_dev': class_df['temperature'].std(),
            'humidity_max': class_df['humidity'].max(),
            'humidity_min': class_df['humidity'].min(),
            'humidity_range': class_df['humidity'].max() - class_df['humidity'].min(),
            'humidity_mean': class_df['humidity'].mean(),
            'humidity_variance': class_df['humidity'].var(),
            'humidity_std_dev': class_df['humidity'].std(),
            'ph_max': class_df['ph'].max(),
            'ph_min': class_df['ph'].min(),
            'ph_range': class_df['ph'].max() - class_df['ph'].min(),
            'ph_mean': class_df['ph'].mean(),
            'ph_variance': class_df['ph'].var(),
            'ph_std_dev': class_df['ph'].std(),
            'rainfall_max': class_df['rainfall'].max(),
            'rainfall_min': class_df['rainfall'].min(),
            'rainfall_range': class_df['rainfall'].max() - class_df['rainfall'].min(),
            'rainfall_mean': class_df['rainfall'].mean(),
            'rainfall_variance': class_df['rainfall'].var(),
            'rainfall_std_dev': class_df['rainfall'].std()
        }

        stats_list.append(stats)

    # Create a DataFrame from the statistics list
    stats_df = pd.DataFrame(stats_list)

    # Save the statistics to an Excel file
    save_path2 = os.path.join(
        results_path,
        'class_statistics.xlsx')
    stats_df.to_excel(
        save_path2,
        index=False, index_label=False)


make_plot_and_stats()
