import pandas as pd
import numpy as np
import random
from correlation import check_feature_correlation
import arff
from itertools import chain
from scipy.stats import mstats
from utilities import fold_dataframe, fold_multiple_times, extend_indices_to_spans
def simulate_abrupt_drift(num_drifts, concept1, concept2, shuffle=False):
    """
    Simulates abrupt drifts by combining shuffled chunks of data from two concepts.

    Parameters:
    - num_drifts (int): Number of drifts to introduce.
    - concept1 (pd.DataFrame, str): First concept (pandas DataFrame or file path).
    - concept2 (pd.DataFrame, str): Second concept (pandas DataFrame or file path).
    - shuffle (bool): Whether to shuffle the data sources before splitting. Default is False.

    Returns:
    - pd.DataFrame: Combined DataFrame with abrupt drifts.
    """

    # Load data if file paths are given
    if isinstance(concept1, str):
        if concept1.endswith('.csv'):
            concept1 = pd.read_csv(concept1)
        elif concept1.endswith('.arff'):
            from scipy.io import arff
            data, meta = arff.loadarff(concept1)
            concept1 = pd.DataFrame(data)
    if isinstance(concept2, str):
        if concept2.endswith('.csv'):
            concept2 = pd.read_csv(concept2)
        elif concept2.endswith('.arff'):
            from scipy.io import arff
            data, meta = arff.loadarff(concept2)
            concept2 = pd.DataFrame(data)

    # Shuffle the data sources if specified
    if shuffle:
        concept1 = concept1.sample(frac=1).reset_index(drop=True)
        concept2 = concept2.sample(frac=1).reset_index(drop=True)

    # Validate the number of drifts
    if num_drifts < 1:
        raise ValueError("The number of drifts must be at least 1.")

    # Calculate the number of chunks needed
    total_chunks = num_drifts + 1
    chunk_size1 = len(concept1) // (total_chunks // 2)
    chunk_size2 = len(concept2) // (total_chunks // 2)

    # Ensure both concepts have enough data
    if chunk_size1 == 0 or chunk_size2 == 0:
        raise ValueError("Concept NZEP_datasets are too small for the number of drifts requested.")

    # Create chunks with alternating concepts
    combined_data = []
    for i in range(total_chunks):
        # if i % 2 == 0:
            combined_data.append(concept1.iloc[i * chunk_size1:(i + 1) * chunk_size1])
        # else:
            combined_data.append(concept2.iloc[i * chunk_size2:(i + 1) * chunk_size2])

    # Concatenate the data
    result = pd.concat(combined_data, ignore_index=True)

    return result



def simulate_gradual_drift(num_drifts, concept1, concept2, shuffle=False, drift_lengths=1000):
    """
    Simulates gradual drifts by creating transitional drift periods between concepts.

    Parameters:
    - num_drifts (int): Number of drifts to introduce.
    - concept1 (pd.DataFrame, str): First concept (pandas DataFrame or file path).
    - concept2 (pd.DataFrame, str): Second concept (pandas DataFrame or file path).
    - shuffle (bool): Whether to shuffle the data sources before splitting. Default is False.
    - drift_lengths (int or list): Length(s) of drifting period (total length 2n). Default is 1000.

    Returns:
    - pd.DataFrame: Combined DataFrame with gradual drifts.
    """

    # Load data if file paths are given
    if isinstance(concept1, str):
        concept1 = pd.read_csv(concept1) if concept1.endswith('.csv') else pd.DataFrame(arff.load(concept1)[0])
    if isinstance(concept2, str):
        concept2 = pd.read_csv(concept2) if concept2.endswith('.csv') else pd.DataFrame(arff.load(concept2)[0])

    # Shuffle data if specified
    if shuffle:
        concept1 = concept1.sample(frac=1).reset_index(drop=True)
        concept2 = concept2.sample(frac=1).reset_index(drop=True)

    # Handle drift lengths
    if isinstance(drift_lengths, int):
        drift_lengths = [drift_lengths] * num_drifts
    if len(drift_lengths) != num_drifts:
        raise ValueError("Length of drift_lengths list must match the number of drifts.")

    combined_data = []

    # Determine chunk sizes and drift handling
    chunk_size = (len(concept1) + len(concept2) - sum(drift_lengths)) // (num_drifts + 1)
    # chunk_size2 = (len(concept2) - sum(drift_lengths) // 2) // (num_drifts + 1)

    current_index1 = 0
    current_index2 = 0

    for i in range(num_drifts):
        # Add concept1 chunk
        concept1_chunk = concept1.iloc[current_index1:current_index1 + chunk_size]
        combined_data.append(concept1_chunk)
        current_index1 += chunk_size

        # Add gradual drift period
        drift_length = drift_lengths[i]
        n = drift_length // 2
        drift_chunk1 = concept1.iloc[current_index1:current_index1 + n]
        drift_chunk2 = concept2.iloc[current_index2:current_index2 + n]
        drift_chunk = pd.concat([drift_chunk1, drift_chunk2]).sample(frac=1).reset_index(drop=True)  # Shuffle
        combined_data.append(drift_chunk)
        current_index1 += n
        current_index2 += n

    # Add remaining concept1 and concept2 chunks
    combined_data.append(concept1.iloc[current_index1:])
    combined_data.append(concept2.iloc[current_index2:])

    # Concatenate the final data
    result = pd.concat(combined_data, ignore_index=True)

    # Check final length
    expected_length = len(concept1) + len(concept2)
    if len(result) != expected_length:
        raise ValueError(f"Unexpected final length: {len(result)} rows instead of {expected_length} rows.")

    return result


# def simulate_incremental_drift(concept, num_drifts, drift_lengths=1000, drifting_feature=None,
#                                target_index=-1, method='pearson', drop_drifting_feature=True):
#     """
#     Simulates incremental drifts using a single concept by sorting the drifting period.
#
#     Parameters:
#     - concept (pd.DataFrame, str): Concept data (pandas DataFrame or file path).
#     - num_drifts (int): Number of drifts to introduce.
#     - drift_lengths (int or list): Length(s) of drifting period. Default is 1000.
#     - drifting_feature (str): Feature on which sorting occurs during drift. Default is None.
#     - target_index (int): Index of the target feature for correlation computation. Default is -1.
#     - method (str): Correlation method ('pearson', 'spearman', etc.). Default is 'pearson'.
#
#     Returns:
#     - pd.DataFrame: Combined DataFrame with incremental drifts.
#     """
#
#     # Load data if file path is given
#     if isinstance(concept, str):
#         if concept.endswith('.csv'):
#             concept = pd.read_csv(concept)
#         elif concept.endswith('.arff'):
#             from scipy.io import arff
#             data, meta = arff.loadarff(concept)
#             concept = pd.DataFrame(data)
#
#     # Determine drifting feature if None
#     if drifting_feature is None:
#         drifting_feature = check_feature_correlation(concept, target_index, method)
#
#     if drifting_feature not in concept.columns:
#         raise ValueError(f"The feature '{drifting_feature}' is not present in the concept.")
#
#     # Handle drift lengths (single value or list)
#     if isinstance(drift_lengths, int):
#         drift_lengths = [drift_lengths] * num_drifts
#     elif isinstance(drift_lengths, list) and len(drift_lengths) != num_drifts:
#         raise ValueError("Length of drift_lengths list must match the number of drifts.")
#
#     # Sort the entire dataset by the drifting feature
#     concept = concept.sort_values(by=drifting_feature).reset_index(drop=True)
#
#     total_drift_length = sum(drift_lengths)
#     remaining_length = len(concept) - total_drift_length
#     if remaining_length <= 0:
#         raise ValueError("Total drift length exceeds dataset size.")
#
#     # Calculate lengths of non-drifting parts
#     num_non_drifting_parts = num_drifts + 1
#     non_drifting_lengths = [remaining_length // num_non_drifting_parts] * num_non_drifting_parts
#
#     chunks = np.array_split(concept, num_drifts * 2 + 1)
#
#     non_drifting_chunks = chunks[::2]
#     drifting_chunks = chunks[1::2]
#     random.shuffle(non_drifting_chunks)
#     drifting_instances = pd.concat(drifting_chunks, ignore_index=True)
#
#
#     drifting_periods = []
#     for i in range(len(non_drifting_chunks) - 1):
#         if chunks[i][drifting_feature].mean() > chunks[i+1][drifting_feature].mean():
#             filtered_df = drifting_instances[(drifting_instances[drifting_feature] >= chunks[i+1][drifting_feature].max()) & (drifting_instances[drifting_feature] <= chunks[i][drifting_feature].min())]
#             selected_df = filtered_df.sample(n=drift_lengths[i], replace=True)
#             drifting_instances.drop(selected_df.index, inplace=True)
#             drifting_periods.append(selected_df.sort_values(by=drifting_feature, ascending=False))
#         else:
#             filtered_df = drifting_instances[(drifting_instances[drifting_feature] >= chunks[i][drifting_feature].max()) & (drifting_instances[drifting_feature] <= chunks[i+1][drifting_feature].min())]
#             selected_df = filtered_df.sample(n=drift_lengths[i], replace=True)
#             drifting_instances.drop(selected_df.index, inplace=True)
#             drifting_periods.append(selected_df.sort_values(by=drifting_feature, ascending=True))
#     drifting_periods.append(drifting_instances)
#
#             # non_drifting_chunks[i] = non_drifting_chunks[i].sort_values(by=drifting_feature, ascending=False)
#
#     combined_data = []
#
#     for i in range(len(drifting_periods)):
#         combined_data.append(pd.DataFrame(non_drifting_chunks[i]).sample(frac=1).reset_index(drop=True))
#         combined_data.append(pd.DataFrame(drifting_periods[i]))
#     combined_data.append(pd.DataFrame(non_drifting_chunks[-1]).sample(frac=1).reset_index(drop=True))
#
#     # for i in range(remaining_length % num_non_drifting_parts):
#     #     non_drifting_lengths[i] += 1  # Distribute remaining rows evenly
#
#     # combined_data = []
#     # current_index = 0
#
#     # for i in range(num_drifts):
#     #     # Add non-drifting part
#     #     non_drift_chunk = concept.iloc[current_index:current_index + non_drifting_lengths[i]]
#     #     non_drift_chunk = non_drift_chunk.sample(frac=1).reset_index(drop=True)
#     #     combined_data.append(non_drift_chunk)
#     #     current_index += non_drifting_lengths[i]
#
#         # Add drifting part
#         # drift_length = drift_lengths[i]
#         # drift_chunk = concept.iloc[current_index:current_index + drift_length]
#         # # ascending = i % 2 == 0  # Alternate between ascending and descending order
#         # drift_chunk = drift_chunk.sort_values(by=drifting_feature, ascending=True).reset_index(drop=True)
#         # combined_data.append(drift_chunk)
#         # current_index += drift_length
#
#     # Add final non-drifting part
#     # non_drift_chunk = concept.iloc[current_index:]
#     # non_drift_chunk = non_drift_chunk.sample(frac=1).reset_index(drop=True)
#     # combined_data.append(non_drift_chunk)
#
#     # Concatenate final DataFrame
#     result = pd.concat(combined_data, ignore_index=True)
#
#     if drop_drifting_feature:
#         result = result.drop(columns=[drifting_feature])
#
#     return result



# def simulate_incremental_drift(concept, num_drifts, drift_lengths=1000, drifting_feature=None,
#                          target_index=-1, method='pearson', drop_drifting_feature=True):
#     """
#     Simulate incremental drifts in the dataset by sequentially sampling from the dataset.
#
#     Parameters:
#     - concept: pd.DataFrame or array-like, original dataset.
#     - num_drifts: int, number of drifts.
#     - drift_lengths: int, length of the incremental drift period.
#     - drifting_feature: str or int, feature used for sorting and simulating drift.
#     - target_index: int, index of the target variable (default -1 for last column).
#     - method: str, optional method for additional processing ('pearson' placeholder).
#     - drop_drifting_feature: bool, whether to drop the drifting feature from the final dataset.
#
#     Returns:
#     - pd.DataFrame, simulated dataset with drift.
#     """
#
#     # Load data if file path is given
#     if isinstance(concept, str):
#         if concept.endswith('.csv'):
#             concept = pd.read_csv(concept)
#         elif concept.endswith('.arff'):
#             from scipy.io import arff
#             data, meta = arff.loadarff(concept)
#             concept = pd.DataFrame(data)
#
#     # Determine drifting feature if None
#     if drifting_feature is None:
#         drifting_feature = check_feature_correlation(concept, target_index, method)
#
#     if drifting_feature not in concept.columns:
#         raise ValueError(f"The feature '{drifting_feature}' is not present in the concept.")
#
#     # Handle drift lengths (single value or list)
#     # if isinstance(drift_lengths, int):
#     #     drift_lengths = [drift_lengths] * num_drifts
#     # elif isinstance(drift_lengths, list) and len(drift_lengths) != num_drifts:
#     #     raise ValueError("Length of drift_lengths list must match the number of drifts.")
#
#     simulated_data = []
#     current_data = concept.sort_values(by=drifting_feature).reset_index(drop=True)
#     total_rows = len(current_data)
#
#     # Calculate the initial concept size
#     initial_concept_size = (total_rows - num_drifts * drift_lengths) // (num_drifts + 1)
#
#     # Start forming the first concept
#     first_concept = current_data.iloc[:initial_concept_size]
#     simulated_data.append(first_concept.sample(frac=1).reset_index(drop=True))
#
#     remaining_data = current_data.iloc[initial_concept_size:].reset_index(drop=True)
#
#     # Create drifts and subsequent concepts
#     for i in range(num_drifts):
#         # Drift phase (ascending or descending)
#         if i % 2 == 0:
#             drift_period = remaining_data.iloc[:drift_lengths]
#         else:
#             drift_period = remaining_data.iloc[:drift_lengths].sort_values(by=drifting_feature, ascending=False)
#         simulated_data.append(drift_period)
#         remaining_data = remaining_data.iloc[drift_lengths:].reset_index(drop=True)
#
#         # Define min and max range for next concept
#         min_value = drift_period[drifting_feature].max()
#         range_diff = first_concept[drifting_feature].max() - first_concept[drifting_feature].min()
#         max_value = min_value + range_diff
#
#         next_concept = remaining_data[(remaining_data[drifting_feature] >= min_value) &
#                                       (remaining_data[drifting_feature] <= max_value)]
#         next_concept = next_concept.sample(n=min(initial_concept_size, len(next_concept)), random_state=42)
#         simulated_data.append(next_concept)
#
#         # Update remaining data
#         remaining_data = remaining_data.drop(next_concept.index).reset_index(drop=True)
#
#     # Concatenate all concepts and drifts
#     final_dataset = pd.concat(simulated_data).reset_index(drop=True)
#
#     # Optionally drop the drifting feature
#     if drop_drifting_feature:
#         final_dataset = final_dataset.drop(columns=[drifting_feature])
#
#     return final_dataset


# def get_quantiles(data, num_quantiles):
#     """
#     Returns the quantile boundaries for the given data.
#
#     Parameters:
#     - data: List or NumPy array of numerical values.
#     - num_quantiles: The number of quantiles to compute (e.g., 4 for quartiles, 10 for deciles).
#
#     Returns:
#     - A dictionary containing quantile ranges.
#     """
#     if not isinstance(num_quantiles, int) or num_quantiles <= 0:
#         raise ValueError("Number of quantiles must be a positive integer.")
#
#     quantiles = np.linspace(0, 1, num_quantiles + 1)  # Create equal probability intervals
#     quantile_values = np.quantile(data, quantiles)  # Compute quantile values
#
#     return [quantile_values[i] for i in range(1, len(quantile_values) - 1)]


# def simulate_incremental_drift(concept, num_drifts, drift_lengths=1000, drifting_feature=None,
#                          target_index=-1, method='pearson', drop_drifting_feature=True):
#     """
#     Simulate incremental drifts in the dataset by sequentially sampling from the dataset.
#
#     Parameters:
#     - concept: pd.DataFrame or array-like, original dataset.
#     - num_drifts: int, number of drifts.
#     - drift_lengths: int, length of the incremental drift period.
#     - drifting_feature: str or int, feature used for sorting and simulating drift.
#     - target_index: int, index of the target variable (default -1 for last column).
#     - method: str, optional method for additional processing ('pearson' placeholder).
#     - drop_drifting_feature: bool, whether to drop the drifting feature from the final dataset.
#
#     Returns:
#     - pd.DataFrame, simulated dataset with drift.
#     """
#
#     # Load data if file path is given
#     if isinstance(concept, str):
#         if concept.endswith('.csv'):
#             concept = pd.read_csv(concept)
#         elif concept.endswith('.arff'):
#             from scipy.io import arff
#             data, meta = arff.loadarff(concept)
#             concept = pd.DataFrame(data)
#
#     # Determine drifting feature if None
#     if drifting_feature is None:
#         drifting_feature = check_feature_correlation(concept, target_index, method)
#
#     if drifting_feature not in concept.columns:
#         raise ValueError(f"The feature '{drifting_feature}' is not present in the concept.")
#
#     # Handle drift lengths (single value or list)
#     # if isinstance(drift_lengths, int):
#     #     drift_lengths = [drift_lengths] * num_drifts
#     # elif isinstance(drift_lengths, list) and len(drift_lengths) != num_drifts:
#     #     raise ValueError("Length of drift_lengths list must match the number of drifts.")
#
#     simulated_data = []
#     current_data = concept.sort_values(by=drifting_feature).reset_index(drop=True)
#     cuts = get_quantiles(current_data[drifting_feature], 2 * num_drifts)
#     cuts.insert(0, current_data[drifting_feature].min())
#     cuts.append(current_data[drifting_feature].max())
#
#     drift_rows = 0
#     if isinstance(drift_lengths, int):
#         drift_lengths = [drift_lengths] * num_drifts
#     drift_rows = sum(drift_lengths)
#
#     chunk_length = (len(current_data) - drift_rows) // (num_drifts + 1)
#
#     # Build first concept with the first quantile
#     first_concept = current_data[current_data[drifting_feature] <= cuts[1]]
#     cuts.remove(cuts[0])
#     simulated_data.append(first_concept.sample(frac=1).reset_index(drop=True))
#     # Build following concept with a random quantile
#     for i in range(2, len(cuts) - 1):
#         random_cut = random.choice(cuts)
#         cuts.remove(random_cut)
#         second_concept = current_data[(current_data[drifting_feature] > cuts[cuts.index(random_cut)]) & (current_data[drifting_feature] <= cuts[cuts.index(random_cut + 1)])].sample(n=chunk_length)
#         # Sample from the rest data to build a incremental period from the first to the second concept
#         incremental_period = current_data[(current_data[drifting_feature] > cuts[1]) & (current_data[drifting_feature] <= cuts[random_cut])].sample(n=drift_lengths[i - 2])
#         if first_concept[drifting_feature].mean() > second_concept[drifting_feature].mean():
#             incremental_period = incremental_period.sort_values(by=drifting_feature, ascending=False)
#         else:
#             incremental_period = incremental_period.sort_values(by=drifting_feature, ascending=True)
#
#         simulated_data.append(incremental_period)
#         simulated_data.append(second_concept.sample(frac=1).reset_index(drop=True))

    # total_rows = len(current_data)
    #
    # # Calculate the initial concept size
    # initial_concept_size = (total_rows - num_drifts * drift_lengths) // (num_drifts + 1)
    #
    # # Start forming the first concept
    # first_concept = current_data.iloc[:initial_concept_size]
    # simulated_data.append(first_concept.sample(frac=1).reset_index(drop=True))
    #
    # remaining_data = current_data.iloc[initial_concept_size:].reset_index(drop=True)
    #
    # # Create drifts and subsequent concepts
    # for i in range(num_drifts):
    #     # Drift phase (ascending or descending)
    #     if i % 2 == 0:
    #         drift_period = remaining_data.iloc[:drift_lengths]
    #     else:
    #         drift_period = remaining_data.iloc[:drift_lengths].sort_values(by=drifting_feature, ascending=False)
    #     simulated_data.append(drift_period)
    #     remaining_data = remaining_data.iloc[drift_lengths:].reset_index(drop=True)
    #
    #     # Define min and max range for next concept
    #     min_value = drift_period[drifting_feature].max()
    #     range_diff = first_concept[drifting_feature].max() - first_concept[drifting_feature].min()
    #     max_value = min_value + range_diff
    #
    #     next_concept = remaining_data[(remaining_data[drifting_feature] >= min_value) &
    #                                   (remaining_data[drifting_feature] <= max_value)]
    #     next_concept = next_concept.sample(n=min(initial_concept_size, len(next_concept)), random_state=42)
    #     simulated_data.append(next_concept)
    #
    #     # Update remaining data
    #     remaining_data = remaining_data.drop(next_concept.index).reset_index(drop=True)

    # Concatenate all concepts and drifts
    # final_dataset = pd.concat(simulated_data).reset_index(drop=True)
    #
    # # Optionally drop the drifting feature
    # if drop_drifting_feature:
    #     final_dataset = final_dataset.drop(columns=[drifting_feature])
    #
    # return final_dataset

def simulate_incremental_drift(concept, num_drifts, drift_lengths=1000, drifting_feature=None,
                         target_index=-1, method='pearson', drop_drifting_feature=True):
    """
    Simulate incremental drifts in the dataset by sequentially sampling from the dataset.

    Parameters:
    - concept: pd.DataFrame or array-like, original dataset.
    - num_drifts: int, number of drifts.
    - drift_lengths: int, length of the incremental drift period.
    - drifting_feature: str or int, feature used for sorting and simulating drift.
    - target_index: int, index of the target variable (default -1 for last column).
    - method: str, optional method for additional processing ('pearson' placeholder).
    - drop_drifting_feature: bool, whether to drop the drifting feature from the final dataset.

    Returns:
    - pd.DataFrame, simulated dataset with drift.
    """

    # Load data if file path is given
    if isinstance(concept, str):
        if concept.endswith('.csv'):
            concept = pd.read_csv(concept)
        elif concept.endswith('.arff'):
            from scipy.io import arff
            data, meta = arff.loadarff(concept)
            concept = pd.DataFrame(data)

    # Determine drifting feature if None
    if drifting_feature is None:
        drifting_feature = check_feature_correlation(concept, target_index, method)

    if drifting_feature not in concept.columns:
        raise ValueError(f"The feature '{drifting_feature}' is not present in the concept.")

    # Handle drift lengths (single value or list)
    if isinstance(drift_lengths, int):
        drift_lengths = [drift_lengths] * num_drifts
    elif isinstance(drift_lengths, list) and len(drift_lengths) != num_drifts:
        raise ValueError("Length of drift_lengths list must match the number of drifts.")

    # Sort the entire dataset by the drifting feature
    concept = concept.sort_values(by=drifting_feature).reset_index(drop=True)

    total_drift_length = sum(drift_lengths)
    remaining_length = len(concept) - total_drift_length
    if remaining_length <= 0:
        raise ValueError("Total drift length exceeds dataset size.")

    # Calculate lengths of non-drifting parts
    num_non_drifting_parts = num_drifts + 1
    non_drifting_lengths = [remaining_length // num_non_drifting_parts] * num_non_drifting_parts
    list_of_parts_length = []
    for i in range(num_drifts):
        list_of_parts_length.append(non_drifting_lengths[i])
        list_of_parts_length.append(drift_lengths[i])
    list_of_parts_length.append(len(concept) - sum(list_of_parts_length))

    if sum(list_of_parts_length) != len(concept):
        raise ValueError("The sum of the lengths of the parts does not match the length of the concept.")

    # Split the concept into parts
    drifting_feature_values = concept[drifting_feature]
    drifting_value_range = drifting_feature_values.max() - drifting_feature_values.min()

    step = drifting_feature_values[int(len(drifting_feature_values) * 0.2)]
    num_concepts = num_drifts + 1

    fold_times = 0
    if num_drifts > 1:
        decider = random.random()
        if decider > 0.5:
            higher = 'left'
            frac = 0.55
        else:
            higher = 'right'
            frac = 0.45
        folded_concept = fold_dataframe(concept, drifting_feature, step, frac, higher, shape='^' if random.random() > 0.5 else 'v')
    elif num_drifts <= 1:
        folded_concept = [concept]
    fold_times += 1

    lst_concept = fold_multiple_times(folded_concept, fold_dataframe, times=num_drifts - 2, column=drifting_feature)
    local_peaks = [0]
    for l in lst_concept:
        local_peaks.append(local_peaks[-1] + len(l))
    # local_peaks.append(-1)

    list_of_spans = extend_indices_to_spans(local_peaks, non_drifting_lengths[0])


    merged_df = pd.concat(lst_concept, ignore_index=True)
    combined = []
    for i in range(len(list_of_spans) - 1):
        chunk = merged_df.iloc[list_of_spans[i]:list_of_spans[i + 1]]
        # merged_df = merged_df.drop(chunk.index)
        if i % 2 == 0:
            chunk = chunk.sample(frac=1).reset_index(drop=True)
        combined.append(chunk)


    result = pd.concat(combined, ignore_index=True)
    if drop_drifting_feature:
        result = result.drop(columns=[drifting_feature])

    return result, drifting_feature
