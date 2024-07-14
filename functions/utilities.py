import pandas as pd
import os


def load_data(file_path: str, header=0) -> pd.DataFrame:
    """Load a CSV or TSV file into a pandas DataFrame."""
    try:
        print(f'Loading data from: {file_path} ...')
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.tsv':
            df = pd.read_csv(file_path, sep='\t', low_memory=False)
        elif file_extension == '.csv':
            df = pd.read_csv(file_path, sep=',', low_memory=False, header=header)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .tsv file.")

        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        raise
    except ValueError as ve:
        print(f"Error: {ve}")
        raise
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic data cleaning and detect inconsistencies."""
    df.replace({'\\N': pd.NA}, inplace=True)

    return df


def save_clean_data(df, output_path):
    """Save cleaned DataFrame to the specified file path."""
    df.to_csv(output_path, index=False)


def filter_by_common_years(df1, df2, df1_year_column='startYear', df2_start_column=4):
    """
    Filter two DataFrames to only include data for the years that are present in both tables.

    Args:
    df1 (pd.DataFrame): First DataFrame containing a year column.
    df2 (pd.DataFrame): Second DataFrame with years as column names starting from the fifth column.
    df1_year_column (str): The name of the year column in df1. Default is 'startYear'.
    df2_start_column (int): The index of the first year column in df2. Default is 4.

    Returns:
    tuple: Filtered df1 and df2 DataFrames.
    """

    # Convert the year column in df1 to integers
    df1[df1_year_column] = pd.to_numeric(df1[df1_year_column], errors='coerce').dropna().astype(int)

    # Extract the year range from df1
    df1_years = df1[df1_year_column].dropna().astype(int)
    min_year_df1, max_year_df1 = df1_years.min(), df1_years.max()

    # Extract the year range from df2 column names, ignoring the last column
    df2_years = [int(col) for col in df2.columns[df2_start_column:-1] if col.isdigit()]
    min_year_df2, max_year_df2 = min(df2_years), max(df2_years)

    # Determine the common year range
    start_year = max(min_year_df1, min_year_df2)
    end_year = min(max_year_df1, max_year_df2)

    if start_year > end_year:
        raise ValueError("No overlapping years between the two DataFrames.")

    # Filter df1 to include only the years in the common range
    filtered_df1 = df1[(df1[df1_year_column] >= start_year) & (df1[df1_year_column] <= end_year)]

    # Filter df2 to include only the columns in the common range, ignoring the last column
    year_columns = [col for col in df2.columns[df2_start_column:-1] if start_year <= int(col) <= end_year]
    filtered_df2 = df2.loc[:, df2.columns[:df2_start_column].tolist() + year_columns]

    return filtered_df1, filtered_df2


def filter_by_user_year_range(df1, df2, user_start_year=None, user_end_year=None, df1_year_column='startYear',
                              df2_start_column=4):
    """
    Further filter two DataFrames based on a user-specified year range.
    """
    if user_start_year is None and user_end_year is None:
        return df1, df2

    # Extract the year range from df1 and df2 for current data
    min_year_df1 = df1[df1_year_column].min()
    max_year_df1 = df1[df1_year_column].max()
    df2_years = [int(col) for col in df2.columns[df2_start_column:-1] if col.isdigit()]
    min_year_df2 = min(df2_years)
    max_year_df2 = max(df2_years)

    # Determine the actual start and end years to filter
    start_year = max(min_year_df1, min_year_df2)
    end_year = min(max_year_df1, max_year_df2)

    if user_start_year is not None:
        start_year = max(start_year, user_start_year)

    if user_end_year is not None:
        end_year = min(end_year, user_end_year)

    if start_year > end_year:
        raise ValueError("The specified range does not overlap with the data range.")

    # Filter df1 to include only the years in the new range
    filtered_df1 = df1[(df1[df1_year_column] >= start_year) & (df1[df1_year_column] <= end_year)]

    # Filter df2 to include only the columns in the new range, ignoring the last column
    year_columns = [col for col in df2.columns[df2_start_column:-1] if start_year <= int(col) <= end_year]
    filtered_df2 = df2.loc[:, df2.columns[:df2_start_column].tolist() + year_columns]

    return filtered_df1, filtered_df2
