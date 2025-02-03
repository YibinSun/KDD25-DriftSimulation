from drifting_functions import simulate_incremental_drift, simulate_incremental_drift
import numpy as np
import pandas as pd


# Generate array
#
# random_array = np.random.uniform(0, 10, 3400)
# random_array1 = np.random.uniform(-2, 2, 3400)

# data = {
#     'feature': random_array.tolist(),
#     'target': random_array1.tolist()
# }

df = pd.read_csv('/Users/spencer/Documents/Datasets/RDatasets/bike.csv')

# Convert to DataFrame
# df = pd.DataFrame(data)

processed_df, feature = simulate_incremental_drift(df, num_drifts=2, drift_lengths=4000, method='pearson', drop_drifting_feature=False)

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Set backend

x = np.arange(0, len(processed_df))
feature_values = processed_df[feature]
target_values = processed_df['cnt']

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 3))  # 1 row, 2 columns

# Left subplot for 'feature'
sns.lineplot(x=x, y=feature_values, linewidth=0.5, ax=axes[0], color='#17becf')
axes[0].set_title('Selected Feature (Normalized Temperature)')
axes[0].set_xlabel('Index')
axes[0].set_ylabel('Values')

# Right subplot for 'target'
sns.lineplot(x=x, y=target_values, linewidth=0.5, ax=axes[1],color='#9467bd')
axes[1].set_title('Target (Bike Rental Count)')
axes[1].set_xlabel('Index')
axes[1].set_ylabel('Values')

# Save and show plot
plt.tight_layout()  # Ensures everything fits without overlap
# plt.savefig('./test_incremental_drift_side_by_side.png')
plt.show()


# data = np.random.uniform(1, 10, size=1000)
# target = np.random.uniform(1, 10, size=1000)
# data = pd.DataFrame(
#     {'value':data,
#         'target':target
#      }
# )
# sorted_data, index = sort_waving(data, 'value',num_peaks=3)
#
# # Plot the result
# sns.lineplot(data=sorted_data, x=sorted_data.index, y="value", linewidth=1, color="dodgerblue", label="Data with Incremental Drifts")
# # plt.plot(sorted_data, label='Multi-Peak Sorted List')
# # plt.title("List Sorted with Multiple Peaks")
# # plt.xlabel("Index")
# plt.show()