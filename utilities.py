import math

import numpy as np
import pandas as pd
from scipy import stats
import arff
from sdv.sampling import Condition
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
def simulate_shifted_distribution(data, mean_shift, output_size=None):
    """
    Estimate the distribution of a list of numbers and simulate a new list
    with the same distribution but with the mean shifted by a given value.

    Parameters:
        data (list or numpy array): Input list of numbers.
        mean_shift (float): The amount by which to shift the mean.
        output_size (int, optional): Length of the simulated list.
                                     If not specified, the length of `data` is used.

    Returns:
        numpy array: Simulated list of numbers with the shifted mean.
    """
    if output_size is None:
        output_size = len(data)

    # Fit to normal distribution
    mean, std = np.mean(data), np.std(data)

    # Create new mean by shifting
    new_mean = mean + mean_shift

    # Generate new data from normal distribution
    simulated_data = np.random.normal(loc=new_mean, scale=std, size=output_size)

    return simulated_data


def generate_conditional_synthetic_data(numbers_list, ctgan_model, condition_name):
    """
    Generate synthetic data points using CTGAN with conditional values.

    Parameters:
    - numbers_list: List of numerical values (for the condition).
    - ctgan_model: Trained CTGAN model.
    - condition_name: Name of the condition column (string).

    Returns:
    - pd.DataFrame: DataFrame of synthetic data points.
    """
    synthetic_data_points = []  # To store individual synthetic rows

    for number in numbers_list:
        # Generate a single synthetic data point for the condition
        condition = [Condition(column_values={condition_name: number}, num_rows=1)]
        synthetic_data = ctgan_model.sample_from_conditions(conditions=condition)

        # Add the generated data point to the list
        synthetic_data_points.append(synthetic_data)

    # Concatenate all generated data points into a single DataFrame
    combined_synthetic_data = pd.concat(synthetic_data_points, ignore_index=True)

    return combined_synthetic_data




def read_arff_to_dataframe(file_path):
    """
    Reads an ARFF file and converts it to a Pandas DataFrame.

    Parameters:
    - file_path (str): Path to the ARFF file.

    Returns:
    - pd.DataFrame: A Pandas DataFrame containing the ARFF file data.
    """
    with open(file_path, 'r') as file:
        arff_data = arff.load(file)

    # Convert to DataFrame
    df = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in arff_data['attributes']])

    return df



def load_data(input_data):
    """
    Load data based on input type.
    """
    if isinstance(input_data, pd.DataFrame):
        return input_data
    elif isinstance(input_data, str):
        if input_data.endswith('.csv'):
            return pd.read_csv(input_data)
        elif input_data.endswith('.arff'):
            data = arff.load(input_data)
            return pd.DataFrame(data[0])
        else:
            raise ValueError("Unsupported file type. Please use a CSV or ARFF file.")
    else:
        raise ValueError("Input data must be a DataFrame, CSV file path, or ARFF file path.")


def train_ctgan_models(input_data, column_name, num_chunks):
    """
    Trains CTGAN models on chunks of data.

    Parameters:
    - input_data: DataFrame, CSV file path, or ARFF file path.
    - column_name: str, column to sort the data by.
    - num_chunks: int, number of chunks to split the data into.

    Returns:
    - list of trained CTGAN models.
    """
    # Load the data
    df = load_data(input_data)

    # Ensure the column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the dataset.")

    # Sort data by the specified column
    df = df.sort_values(by=column_name).reset_index(drop=True)

    # Split data into chunks
    chunk_size = len(df) // num_chunks
    chunks = [df.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

    # Handle remaining data in case of uneven split
    if len(df) % num_chunks != 0:
        chunks[-1] = pd.concat([chunks[-1], df.iloc[num_chunks * chunk_size:]], axis=0)

    # Train CTGAN models for each chunk
    models = []
    for chunk in chunks:
        metadata = Metadata()
        metadata = metadata.detect_from_dataframe(chunk)
        model = CTGANSynthesizer(metadata)
        model.fit(chunk)
        models.append(model)

    return models


def fold_dataframe(df, column, difference: float, frac: float,
                   higher='left' # or 'right'
                   , shape: str = '^' # or 'v'
 ):
    """
    Folds a DataFrame into two parts based on a difference in values.

    Parameters:
    - df: pandas DataFrame
    - column: the column to sort and fold by
    - difference: the difference in values that separates the two parts
    - frac: the fraction of the DataFrame to use in the first part

    Returns:
    - combined: folded DataFrame with one side reaching the difference and the other closer to min/max
    """
    # Calculate the cutoff value

    copy = df.copy()
    copy = copy.sort_values(column, ascending=True)
    feature_values = copy[column]

    if shape == '^' and higher == 'left':
        asymmetric = copy[copy[column] <= feature_values.min() + difference]
        symmetric = copy.drop(asymmetric.index)
        asymmetric_ratio = asymmetric.shape[0]/copy.shape[0]
        left_ratio = frac - asymmetric_ratio

        left = symmetric.sample(frac=left_ratio).sort_values(column, ascending=True)
        right = symmetric.drop(left.index).sort_values(column, ascending=False)

        merged = pd.concat([asymmetric, left]).sort_values(column, ascending=True)

        rebuilt = [merged, right]

    elif shape == '^' and higher == 'right':
        asymmetric = copy[copy[column] <= feature_values.min() + difference]
        symmetric = copy.drop(asymmetric.index)
        asymmetric_ratio = asymmetric.shape[0]/copy.shape[0]
        right_ratio = 1 - frac - asymmetric_ratio

        right = symmetric.sample(frac=right_ratio).sort_values(column, ascending=False)
        left = symmetric.drop(right.index).sort_values(column, ascending=True)

        merged = pd.concat([asymmetric, right]).sort_values(column, ascending=False)
        rebuilt = [left, merged]

    elif shape == 'v' and higher == 'left':
        asymmetric = copy[copy[column] >= feature_values.max() - difference]
        symmetric = copy.drop(asymmetric.index)
        asymmetric_ratio = asymmetric.shape[0]/copy.shape[0]
        left_ratio = frac - asymmetric_ratio

        left = symmetric.sample(frac=left_ratio).sort_values(column, ascending=False)
        right = symmetric.drop(left.index).sort_values(column, ascending=True)

        merged = pd.concat([left, asymmetric]).sort_values(column, ascending=False)
        rebuilt = [merged, right]

    else:
        asymmetric = copy[copy[column] >= feature_values.max() - difference]
        symmetric = copy.drop(asymmetric.index)
        asymmetric_ratio = asymmetric.shape[0]/copy.shape[0]
        right_ratio = 1 - frac - asymmetric_ratio

        right = symmetric.sample(frac=right_ratio).sort_values(column, ascending=True)
        left = symmetric.drop(right.index).sort_values(column, ascending=False)

        merged = pd.concat([right, asymmetric]).sort_values(column, ascending=True)
        rebuilt = [left, merged]

    return rebuilt

def first_half(index, lst):
    # check if the index is in the first half of the list
    return index <= len(lst) // 2


def fold_multiple_times(df, func, times, column):
    if times <= 0:
        return [df] if isinstance(df, pd.DataFrame) else df

    if isinstance(df, pd.DataFrame):
        dfs = [df]
    else:
        dfs = list(df)

    result = []
    repetition = 0
    for index, sub_df in enumerate(dfs):
        difference = (sub_df[column].max() - sub_df[column].min()) * 0.2
        # difference = sub_df[column][int(len(sub_df) * 0.2)]
        if (len(dfs)/2) % 2 != 0:
            result.extend(
                func(sub_df, frac=0.55, higher='right' if first_half(index, dfs) else 'left', shape='v', difference=difference,
                     column=column))
            repetition += 1
            if repetition >= (times ):
                result.extend(dfs[index + 1:])
                break
        else:
            result.extend(
                func(sub_df, frac=0.55, higher='left' if first_half(index, dfs) else 'right', shape='^', difference=difference,
                     column=column))
            repetition += 1
            if repetition >= (times ):
                result.extend(dfs[index + 1:])
                break
        # if first_half(index, dfs):
        #     if index % 2 == 0:
        #         shape=''
        #     result.extend(func(sub_df, frac=0.5, higher='right', shape='v' if index % 2 == 0 else '^', difference=difference, column=column))
        #     repetition += 1
        #     if repetition >= (times - 1):
        #         result.extend(dfs[index+1:])
        #         break
        # else:
        #     result.extend(func(sub_df, frac=0.5, higher='left', shape='v' if index % 2 == 0 else 'v', difference=difference, column=column))
        #     repetition += 1
        #     if repetition >= (times - 1):
        #         result.extend(dfs[index+1:])
        #         break

    return fold_multiple_times(result, fold_dataframe, times-repetition, column)


def extend_indices_to_spans(indices, length):
    spans = []
    n = len(indices)

    for i, index in enumerate(indices):
        if i == 0:
            # First index: add an end
            start = index
            end = index + length
        elif i >= n - 1:
            # Last index: add a start
            start = index - length
            end = index
        else:
            # Middle index: center the span
            start = index - length // 2
            end = index + length // 2

        spans.append(max(0, start))  # Ensure the span doesn’t go below 0.
        spans.append(end)
    return spans
