import pandas as pd
from scipy.io import arff
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor


def check_feature_correlation(data, target_index=-1, method='pearson'):
    """
    Function to check feature correlation with the target.

    Parameters:
        data (str or pd.DataFrame): Path to CSV/ARFF file or a pandas DataFrame.
        target_index (int): Index of the target column (-1 for last column).
        method (str): Correlation method ('pearson', 'spearman', 'mutual_info', 'feature_importance').

    Returns:
        list: Names of the most correlated features (or indices if no column names).
    """

    # Load CSV or ARFF if file path is provided
    if isinstance(data, str):
        if data.endswith('.csv'):
            df = pd.read_csv(data)
        elif data.endswith('.arff'):
            raw_data = arff.loadarff(data)
            df = pd.DataFrame(raw_data[0])
            # Decode byte strings if necessary
            df = df.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)
        else:
            raise ValueError("File format not supported. Use .csv or .arff")
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input must be a file path (CSV/ARFF) or a pandas DataFrame.")

    # Handle target index
    if target_index == -1:
        target_index = len(df.columns) - 1
    target_column = df.columns[target_index]

    # Extract features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Perform correlation analysis
    if method == 'pearson':
        correlation = X.corrwith(y, method='pearson').sort_values(ascending=False)
    elif method == 'spearman':
        correlation = X.corrwith(y, method='spearman').sort_values(ascending=False)
    elif method == 'mutual_info':
        mi_scores = mutual_info_regression(X, y)
        correlation = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    elif method == 'feature_importance':
        model = RandomForestRegressor(random_state=0)
        model.fit(X, y)
        importances = model.feature_importances_
        correlation = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    else:
        raise ValueError("Invalid method. Choose from 'pearson', 'spearman', 'mutual_info', or 'feature_importance'.")

    # Get top correlated feature(s)
    top_features = correlation.index.tolist()[0]
    print(f"Top features by '{method}' correlation: {top_features}")

    return top_features
