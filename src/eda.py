# src/eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def data_summary(df: pd.DataFrame):
    """
    Produces summary statistics for a given DataFrame.
    
    :param df: The DataFrame to be summarized.
    """
    print("Data Summary:\n")
    print(df.describe())
    print("\nData Structure:\n")
    print(df.info())

def univariate_analysis(df: pd.DataFrame):
    """
    Conducts univariate analysis with histograms for numerical variables 
    and bar charts for categorical variables. 
    Displays all charts in a single figure using multiple subplots for enhanced comparison.
    
    :param df: The DataFrame to be analyzed.
    """
    # Identify numerical and categorical columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Calculate the total number of plots
    num_numerical = len(numerical_columns)
    num_categorical = len(categorical_columns)
    total_plots = num_numerical + num_categorical
    
    # Calculate rows and columns needed for subplots
    num_cols = 4  # Define 4 columns per row
    num_rows = (total_plots + num_cols - 1) // num_cols  # Round up to accommodate all plots
    
    # Create a figure with a defined number of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5), constrained_layout=True)
    axes = axes.flatten()  # Flatten axes array for iteration
    
    # Create histograms for numerical columns
    for i, col in enumerate(numerical_columns):
        axes[i].hist(df[col].dropna(), bins=30, edgecolor='black', color='skyblue')
        axes[i].set_title(f'Histogram of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    
    # Create bar charts for categorical columns
    for j, col in enumerate(categorical_columns, start=len(numerical_columns)):
        sns.countplot(data=df, x=col, ax=axes[j], palette='pastel', hue=None, legend=False)
        axes[j].set_title(f'Bar Chart of {col}')
        axes[j].set_xlabel(col)
        axes[j].set_ylabel('Count')
    
    # Hide any unused subplots
    for k in range(total_plots, len(axes)):
        axes[k].set_visible(False)
    
    # Display the plot
    plt.show()


def preprocess_data(df: pd.DataFrame):
    """
    Cleans and prepares the DataFrame by renaming columns, converting date columns, 
    and aggregating the data.
    """
    # Ensure 'TransactionMonth' exists and rename it to 'Date'
    if 'TransactionMonth' not in df.columns:
        raise ValueError("The DataFrame must have a 'TransactionMonth' column.")
    
    df.rename(columns={'TransactionMonth': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Aggregate the data by summing up the TotalPremium and TotalClaims
    df = df.groupby('Date').agg({
        'TotalPremium': 'sum',
        'TotalClaims': 'sum'
    }).reset_index()

    # Set 'Date' as the index for subsequent operations
    df.set_index('Date', inplace=True)
    
    # Resample the data monthly using Month End frequency
    monthly_data = df.resample('M').agg({
        'TotalPremium': 'sum',
        'TotalClaims': 'sum'
    })
    
    # Fill in any missing values using forward fill
    monthly_data.ffill(inplace=True)

    # Check for any missing values after forward fill
    missing_values = monthly_data[['TotalPremium', 'TotalClaims']].isnull().sum()
    if missing_values.any():
        print("Missing values detected after filling:")
        print(missing_values)

    # Compute monthly variations
    monthly_data['MonthlyTotalPremiumChange'] = monthly_data['TotalPremium'].pct_change()
    monthly_data['MonthlyTotalClaimsChange'] = monthly_data['TotalClaims'].pct_change()
    
    # Check for any missing values in the change columns
    missing_changes = monthly_data[['MonthlyTotalPremiumChange', 'MonthlyTotalClaimsChange']].isnull().sum()
    if missing_changes.any():
        print("Missing values found in MonthlyTotalPremiumChange or MonthlyTotalClaimsChange after calculations:")
        print(missing_changes)
    
    # Reset index to bring 'Date' back as a regular column
    monthly_data.reset_index(inplace=True)
    
    # Debugging: Print the column names to verify the changes applied
    print("Columns after data preprocessing:")
    print(monthly_data.columns)

    return monthly_data

def bivariate_analysis(df, total_premium_column, total_claims_column, postal_code_column):
    """
    Analyzes the relationships between Total Premium and Total Claims in relation to PostalCode 
    using scatter plots and correlation matrices.

    :param df: DataFrame containing the relevant data
    :param total_premium_column: Column name for total premium
    :param total_claims_column: Column name for total claims
    :param postal_code_column: Column name for postal codes
    """
    # Verify presence of required columns
    required_columns = [total_premium_column, total_claims_column, postal_code_column]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the DataFrame.")

    # Step 1: Group by PostalCode and compute the mean of TotalPremium and TotalClaims
    postal_code_groups = df.groupby(postal_code_column)[[total_premium_column, total_claims_column]].mean().reset_index()

    # Step 2: Create visual representations
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Scatter plot of TotalPremium against TotalClaims by PostalCode
    sns.scatterplot(
        x=total_premium_column, 
        y=total_claims_column, 
        hue=postal_code_column, 
        data=postal_code_groups, 
        palette='viridis', 
        s=100, 
        ax=axes[0]
    )
    axes[0].set_title(f'{total_premium_column} vs {total_claims_column} by {postal_code_column}', fontsize=14)
    axes[0].set_xlabel(f'{total_premium_column}', fontsize=12)
    axes[0].set_ylabel(f'{total_claims_column}', fontsize=12)

    # Generate correlation heatmap between TotalPremium and TotalClaims
    correlation = postal_code_groups[[total_premium_column, total_claims_column]].corr()
    sns.heatmap(
        correlation, 
        annot=True, 
        cmap='coolwarm', 
        fmt='.2f', 
        vmin=-1, 
        vmax=1, 
        center=0, 
        ax=axes[1]
    )
    axes[1].set_title(f'Correlation Matrix: {total_premium_column} & {total_claims_column}', fontsize=14)

    plt.tight_layout()
    plt.show()

def compare_data(df):
    """
    Analyzes trends in insurance cover types, premiums, etc., across different geographic regions using the 'Province' column.
    """
    # Remove leading and trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Confirm the existence of the 'Province' column
    if 'Province' not in df.columns:
        raise ValueError("The DataFrame must include a 'Province' column.")
    
    # Check that the 'Province' column is of type object (string)
    if df['Province'].dtype != 'object':
        raise ValueError("The 'Province' column must be of type 'object' (string).")

    # Verify presence of 'TransactionMonth' column
    if 'TransactionMonth' not in df.columns:
        raise ValueError("The 'TransactionMonth' column is absent from the DataFrame.")
    
    # Ensure 'TransactionMonth' is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['TransactionMonth']):
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')

    # Aggregate total premiums by 'Province' and 'TransactionMonth'
    geo_trends = df.groupby(['Province', pd.Grouper(key='TransactionMonth', freq='M')])['TotalPremium'].sum().unstack()
    
    # Plotting the premium trends
    plt.figure(figsize=(12, 6))
    for province in geo_trends.columns:
        plt.plot(geo_trends.index, geo_trends[province], label=province)

    plt.title('Trends in Total Premiums by Province')
    plt.xlabel('Month')
    plt.ylabel('Total Premium')
    plt.legend()
    plt.grid(True)
    plt.show()

def detect_outliers(df):
    """
    Identifies outliers in numerical data using box plots.
    """
    numeric_columns = df.select_dtypes(include=['number']).columns

    plt.figure(figsize=(14, 7))
    for i, column in enumerate(numeric_columns, 1):
        plt.subplot(3, 4, i)
        sns.boxplot(y=df[column])
        plt.title(f'Boxplot for {column}')

    plt.tight_layout()
    plt.show()

def descriptive_statistics(df: pd.DataFrame):
    """
    Computes and returns the descriptive statistics for numerical columns.

    :param df: DataFrame containing the data to analyze
    :return: DataFrame with descriptive statistics for numerical features
    """
    # Select only numerical columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Calculate descriptive statistics for numerical columns
    numerical_stats = numeric_df.describe().T
    
    # Include additional statistics such as variance and standard deviation
    numerical_stats['variance'] = numeric_df.var()
    numerical_stats['std_dev'] = numeric_df.std()
    
    return numerical_stats

# src/eda.py
def check_data_structure(df: pd.DataFrame):
    """
    Verifies the data types of columns within the DataFrame and ensures that categorical variables and dates are formatted correctly.

    :param df: DataFrame containing the data to evaluate
    :return: Dictionary with column names and their corresponding data types
    """
    # Obtain the data types of each column
    column_dtypes = df.dtypes
    
    # Identify categorical variables (generally of object type)
    categorical_columns = df.select_dtypes(include=['object']).columns
    datetime_columns = df.select_dtypes(include=['datetime']).columns

    # Compile structure information
    data_structure_report = {
        'column_dtypes': column_dtypes,
        'categorical_columns': categorical_columns,
        'datetime_columns': datetime_columns
    }

    return data_structure_report