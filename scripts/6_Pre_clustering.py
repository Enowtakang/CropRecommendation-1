import os
import pandas as pd

"""
1. Paths
"""
data_path = "C:/Users/HP/PycharmProjects/CropRec/data/crop_recommendation.csv"
results_path = "C:/Users/HP/PycharmProjects/CropRec/data/"


"""
Make new data
"""
df = pd.read_csv(data_path)

# Group by the 'label' column and calculate the mean
# for each group
grouped_df = df.groupby('label').mean().reset_index()
# Save the new dataset to a CSV file

save_path = os.path.join(
        results_path, 'averaged_dataset.csv')
grouped_df.to_csv(save_path, index=False)
