# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the data
# data_path = '../data/insurance.csv'
# data = pd.read_csv(data_path)

# # Create a directory to store plots
# import os
# if not os.path.exists('plots'):
#     os.makedirs('plots')

# # Univariate Analysis
# # Numerical Columns: Plot histograms
# numerical_columns = ['Total_Claim', 'Premium']
# for column in numerical_columns:
#     plt.figure(figsize=(8, 5))
#     sns.histplot(data[column], kde=True, bins=20, color='blue')
#     plt.title(f'Distribution of {column}')
#     plt.xlabel(column)
#     plt.ylabel('Frequency')
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     plt.savefig(f'plots/histogram_{column}.png')
#     plt.show()

# # Categorical Columns: Plot bar charts
# categorical_columns = ['Province', 'Zipcode', 'Gender']
# for column in categorical_columns:
#     plt.figure(figsize=(8, 5))
#     data[column].value_counts().plot(kind='bar', color='orange', edgecolor='black')
#     plt.title(f'Distribution of {column}')
#     plt.xlabel(column)
#     plt.ylabel('Count')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.savefig(f'plots/bar_chart_{column}.png')
#     plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import shap

import os

def load_data(file_path):
    """
    Load the dataset from the specified file path.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def create_directory(directory):
    """
    Create a directory if it does not already exist.

    Args:
        directory (str): Path of the directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_numerical_distributions(data, columns, output_dir):
    """
    Plot histograms for numerical columns.

    Args:
        data (pd.DataFrame): The dataset.
        columns (list): List of numerical column names.
        output_dir (str): Directory to save the plots.
    """
    for column in columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(data[column], kde=True, bins=20, color='blue')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/histogram_{column}.png')
        plt.show()

def plot_categorical_distributions(data, columns, output_dir):
    """
    Plot bar charts for categorical columns.

    Args:
        data (pd.DataFrame): The dataset.
        columns (list): List of categorical column names.
        output_dir (str): Directory to save the plots.
    """
    for column in columns:
        plt.figure(figsize=(8, 5))
        data[column].value_counts().plot(kind='bar', color='orange', edgecolor='black')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/bar_chart_{column}.png')
        plt.show()
        
def plot_correlation_matrix(data, columns, output_dir):
    """
    Plot the correlation matrix for a set of columns.
    
    Args:
        data (pd.DataFrame): The dataset.
        columns (list): List of numerical columns to include in the correlation matrix.
        output_dir (str): Directory to save the plot.
    """
    correlation = data[columns].corr()
    plt.figure(figsize=(10, 7))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix.png')
    plt.show()

# Function to plot scatter plot for TotalPremium vs TotalClaims by Zipcode
def plot_scatter_by_zipcode(data, output_dir):
    """
    Plot a scatter plot of TotalPremium vs TotalClaims, colored by Zipcode.
    
    Args:
        data (pd.DataFrame): The dataset.
        output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='Total_Claim', y='Premium', hue='Zipcode', palette='viridis', alpha=0.7)
    plt.title('Scatter Plot: TotalPremium vs TotalClaims by Zipcode')
    plt.xlabel('Total Claims')
    plt.ylabel('Premium')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatter_totalpremium_vs_totalclaims.png')
    plt.show()

# Function to plot trends over geography
def plot_trends_over_geography(data, output_dir):
    """
    Plot trends in insurance coverage, premium, or auto make by Province.
    
    Args:
        data (pd.DataFrame): The dataset.
        output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Province', y='Premium', data=data, palette='Set2')
    plt.title('Premium Distribution by Province')
    plt.xlabel('Province')
    plt.ylabel('Premium')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/premium_by_province.png')
    plt.show()

# Function to plot box plot for outlier detection
def plot_boxplot_for_outliers(data, columns, output_dir):
    """
    Plot box plots to detect outliers in numerical columns.
    
    Args:
        data (pd.DataFrame): The dataset.
        columns (list): List of numerical column names.
        output_dir (str): Directory to save the plots.
    """
    for column in columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data[column], color='lightblue')
        plt.title(f'Box Plot of {column}')
        plt.ylabel(column)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/boxplot_{column}.png')
        plt.show()

# # Task 3: A/B Hypothesis Testing
# def ab_hypothesis_testing(data, group_col, test_col):
#     """
#     Perform A/B Hypothesis Testing.
    
#     Parameters:
#     - data: DataFrame containing the data.
#     - group_col: Column name indicating group (A/B).
#     - test_col: Column name of the metric being tested.
#     """
#     # Separate groups
#     group_a = data[data[group_col] == 'A'][test_col]
#     group_b = data[data[group_col] == 'B'][test_col]
    
#     # Perform t-test for numerical data
#     t_stat, p_value = ttest_ind(group_a, group_b)
#     print(f"T-statistic: {t_stat}, p-value: {p_value}")
#     return t_stat, p_value

# Assuming the previous parts of the function are unchanged
def ab_hypothesis_testing(data, group_column, target_column, min_sample_size=30):
    # Group the data by the specified column
    groups = data.groupby(group_column)
    
    # Ensure both groups have enough samples
    group_keys = list(groups.groups.keys())  # Convert dict_keys to list
    if len(group_keys) < 2:
        return {'Error': 'Not enough groups to perform the test.'}
    
    group_a = groups.get_group(group_keys[0])
    group_b = groups.get_group(group_keys[1])
    
    # Ensure the groups meet the minimum sample size
    if len(group_a) < min_sample_size or len(group_b) < min_sample_size:
        return {'Error': 'One or both groups have insufficient sample size.'}
    
    # Perform the t-test for the two groups
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(group_a[target_column], group_b[target_column])
    
    # Calculate means for each group
    group_a_mean = group_a[target_column].mean()
    group_b_mean = group_b[target_column].mean()
    
    # Check if the p-value is less than 0.05 to reject the null hypothesis
    reject_null = p_value < 0.05
    
    return {
        'Group A Mean': group_a_mean,
        'Group B Mean': group_b_mean,
        'T-statistic': t_stat,
        'P-value': p_value,
        'Reject Null Hypothesis': reject_null
    }
# Task 4: Statistical Modeling


def handle_missing_data(data, strategy="mean", columns=None):
    """
    Handle missing data by imputing or removing missing values.

    Args:
        data (pd.DataFrame): The dataset.
        strategy (str): The imputation strategy ('mean', 'median', 'mode', or 'drop').
        columns (list): List of columns to apply the imputation to. If None, applies to all columns.

    Returns:
        pd.DataFrame: Data with missing values handled.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    if strategy not in ["mean", "median", "mode", "drop"]:
        raise ValueError(
            "Invalid strategy. Must be one of ['mean', 'median', 'mode', 'drop']."
        )

    if columns is None:
        columns = data.columns

    # Ensure the specified columns exist in the DataFrame
    if not all(col in data.columns for col in columns):
        raise ValueError("Some columns specified are not in the DataFrame.")

    # Separate numeric and non-numeric columns
    numeric_columns = data[columns].select_dtypes(include=["number"]).columns
    non_numeric_columns = [col for col in columns if col not in numeric_columns]

    # Handle numeric columns based on the strategy
    if strategy == "mean":
        data[numeric_columns] = data[numeric_columns].fillna(
            data[numeric_columns].mean()
        )
    elif strategy == "median":
        data[numeric_columns] = data[numeric_columns].fillna(
            data[numeric_columns].median()
        )
    elif strategy == "mode":
        data[numeric_columns] = data[numeric_columns].fillna(
            data[numeric_columns].mode().iloc[0]
        )
    elif strategy == "drop":
        data = data.dropna(subset=columns)

    # Handle non-numeric columns (for example, fill with mode or drop rows)
    if strategy == "mode":
        data[non_numeric_columns] = data[non_numeric_columns].fillna(
            data[non_numeric_columns].mode().iloc[0]
        )

    print("Missing data handled using strategy:", strategy)
    return data


def encode_categorical_data(data, categorical_columns):
    """
    Perform one-hot encoding for categorical columns.

    Args:
        data (pd.DataFrame): The dataset.
        categorical_columns (list): List of categorical column names.

    Returns:
        pd.DataFrame: Data with categorical features encoded.
    """
    data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    print("Categorical data encoded.")
    return data_encoded


def split_data(data, target_column, test_size=0.3):
    """
    Split the dataset into training and testing sets.

    Args:
        data (pd.DataFrame): The dataset.
        target_column (str): The target column name.
        test_size (float): The proportion of data to be used for testing.

    Returns:
        X_train, X_test, y_train, y_test: The split data.
    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(
        f"Data split into {1 - test_size:.0%} training and {test_size:.0%} testing sets."
    )
    return X_train, X_test, y_train, y_test


# def train_linear_regression(X_train, y_train):
#     """
#     Train a Linear Regression model.

#     Args:
#         X_train (pd.DataFrame): Training data features.
#         y_train (pd.Series): Training data target.

#     Returns:
#         model (LinearRegression): Trained model.
#     """


#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     print("Linear Regression model trained.")
#     return model


# def train_random_forest(X_train, y_train):
#     """
#     Train a Random Forest model.

#     Args:
#         X_train (pd.DataFrame): Training data features.
#         y_train (pd.Series): Training data target.

#     Returns:
#         model (RandomForestRegressor): Trained model.
#     """

#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     print("Random Forest model trained.")
#     return model


# def train_xgboost(X_train, y_train):
#     """
#     Train an XGBoost model.

#     Args:
#         X_train (pd.DataFrame): Training data features.
#         y_train (pd.Series): Training data target.

#     Returns:
#         model (XGBRegressor): Trained model.
#     """
#     model = XGBRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     print("XGBoost model trained.")
#     return model


# def evaluate_model(model, X_test, y_test):
#     """
#     Evaluate a trained model on test data using Mean Squared Error (MSE).

#     Args:
#         model: The trained model.
#         X_test (pd.DataFrame): Test data features.
#         y_test (pd.Series): Test data target.

#     Returns:
#         float: Mean Squared Error of the model.
#     """
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     print(f"Model evaluation complete. Mean Squared Error: {mse:.4f}")
#     return mse


# def feature_importance_analysis(model, X_train):
#     """
#     Analyze the feature importance using the trained model.

#     Args:
#         model: The trained model.
#         X_train (pd.DataFrame): Training data features.

#     Returns:
#         pd.Series: Feature importance scores.
#     """
#     if hasattr(model, "feature_importances_"):
#         feature_importance = model.feature_importances_
#     else:
#         print("Model does not support feature importance analysis.")
#         return None

#     feature_names = X_train.columns
#     importance_df = pd.DataFrame(
#         {"Feature": feature_names, "Importance": feature_importance}
#     ).sort_values(by="Importance", ascending=False)

#     print("Feature importance analysis complete.")
#     return importance_df


# def interpret_model_predictions(model, X_train):
#     """
#     Interpret the model's predictions using SHAP (SHapley Additive exPlanations).

#     Args:
#         model: The trained model.
#         X_train (pd.DataFrame): Training data features.

#     Returns:
#         shap_values: SHAP values for the model's predictions.
#     """
#     explainer = shap.Explainer(model, X_train)
#     shap_values = explainer(X_train)

#     shap.summary_plot(shap_values, X_train, plot_type="bar")
#     print("Model interpretability complete using SHAP.")
#     return shap_values

def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.

    Args:
        X_train (pd.DataFrame): Training data features.
        y_train (pd.Series): Training data target.

    Returns:
        model (LinearRegression): Trained model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression model trained.")
    return model


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model.

    Args:
        X_train (pd.DataFrame): Training data features.
        y_train (pd.Series): Training data target.

    Returns:
        model (RandomForestRegressor): Trained model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Random Forest model trained.")
    return model


def train_xgboost(X_train, y_train):
    """
    Train an XGBoost model.

    Args:
        X_train (pd.DataFrame): Training data features.
        y_train (pd.Series): Training data target.

    Returns:
        model (XGBRegressor): Trained model.
    """
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("XGBoost model trained.")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data using Mean Squared Error (MSE).

    Args:
        model: The trained model.
        X_test (pd.DataFrame): Test data features.
        y_test (pd.Series): Test data target.

    Returns:
        float: Mean Squared Error of the model.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model evaluation complete. Mean Squared Error: {mse:.4f}")
    return mse


def feature_importance_analysis(model, X_train):
    """
    Analyze the feature importance using the trained model.

    Args:
        model: The trained model.
        X_train (pd.DataFrame): Training data features.

    Returns:
        pd.Series: Feature importance scores.
    """
    if hasattr(model, "feature_importances_"):
        feature_importance = model.feature_importances_
    else:
        print("Model does not support feature importance analysis.")
        return None

    feature_names = X_train.columns
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importance}
    ).sort_values(by="Importance", ascending=False)

    print("Feature importance analysis complete.")
    return importance_df


def interpret_model_predictions(model, X_train):
    """
    Interpret the model's predictions using SHAP (SHapley Additive exPlanations).

    Args:
        model: The trained model.
        X_train (pd.DataFrame): Training data features.

    Returns:
        shap_values: SHAP values for the model's predictions.
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    shap.summary_plot(shap_values, X_train, plot_type="bar")
    print("Model interpretability complete using SHAP.")
    return shap_values