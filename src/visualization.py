# # src/visualization.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def outlier_detection(df: pd.DataFrame):
    """
    Identifies and visualizes outliers using box plots within a single figure with subplots.

    :param df: DataFrame containing numerical columns
    """
    # Select numeric columns from the DataFrame
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Determine the layout for subplots (rows and columns)
    num_plots = len(numeric_cols)
    rows = (num_plots // 3) + (num_plots % 3 > 0)  # Calculate the required number of rows
    cols = 3  # Define the number of columns to 3 for optimal arrangement
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()  # Flatten the array of axes for easier indexing
    
    # Iterate over each numeric column and create a box plot for each
    for i, col in enumerate(numeric_cols):
        # Remove null values prior to plotting
        cleaned_data = df[col].dropna()
        sns.boxplot(y=cleaned_data, ax=axes[i])  # Utilize 'y' for single-column boxplots
        axes[i].set_title(f'Outlier Detection for {col}')
    
    # Deactivate any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def compare_trends_over_geography(df: pd.DataFrame, geography_column: str):
    """
    Analyzes and contrasts trends of various features (insurance cover type, premium, auto make) 
    across different geographical regions.

    :param df: DataFrame containing relevant data
    :param geography_column: The geographical column to group data by (e.g., 'Country', 'Province')
    """
    # Plot 1: Analyze the distribution of Insurance Cover Type across geographical regions
    plt.figure(figsize=(12, 8))
    sns.countplot(x=geography_column, hue='CoverType', data=df)
    plt.title(f'Comparison of Insurance Cover Type Over {geography_column}')
    plt.xticks(rotation=45)
    plt.show()

    # Plot 2: Examine the distribution of Total Premium across geographical regions
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=geography_column, y='TotalPremium', data=df)
    plt.title(f'Comparison of Total Premium Over {geography_column}')
    plt.xticks(rotation=45)
    plt.show()

    # Plot 3: Explore the distribution of Auto Make across geographical regions
    plt.figure(figsize=(12, 8))
    sns.countplot(x=geography_column, hue='make', data=df, order=df['make'].value_counts().index)
    plt.title(f'Comparison of Auto Make Over {geography_column}')
    plt.xticks(rotation=45)
    plt.show()

    # Optional: Analyze Total Premium trends over time if the column exists
    if 'TransactionMonth' in df.columns:
        plt.figure(figsize=(14, 8))
        sns.lineplot(x='TransactionMonth', y='TotalPremium', hue=geography_column, data=df)
        plt.title(f'Trend of Total Premium Over Time by {geography_column}')
        plt.xticks(rotation=45)
        plt.show()

def visualize_eda(df: pd.DataFrame):
    """
    Generates three informative and visually appealing plots to highlight the key findings from EDA.

    :param df: DataFrame containing the EDA results
    """
    # Example visualizations (these can be customized based on your analysis)

    # 1. Visualize the distribution of Total Premium
    plt.figure(figsize=(12, 8))
    sns.histplot(df['TotalPremium'], kde=True)
    plt.title('Distribution of Total Premium')
    plt.show()

    # 2. Generate a Correlation Matrix for numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # 3. Plot Total Premium against Total Claims
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=df['TotalPremium'], y=df['TotalClaims'])
    plt.title('Total Premium vs Total Claims')
    plt.show()

