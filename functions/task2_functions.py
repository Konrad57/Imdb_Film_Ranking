import numpy as np
import pandas as pd


def total_votes_by_country(movies_df):
    """
    Calculate the total sum of votes for each country and return a DataFrame.

    Parameters:
    movies_df (pd.DataFrame): DataFrame of movies with country and composite score information.

    Returns:
    pd.DataFrame: DataFrame with two columns: 'country' and 'number of votes'.
    """

    votes_by_country = (movies_df.groupby('country')['numVotes'].sum().reset_index(name='number of votes'))
    votes_by_country.sort_values(by='number of votes', ascending=False, inplace=True)

    return votes_by_country


def average_composite_score_by_country(movies_df):
    """
    Calculate the average composite score for each country and return a DataFrame.

    Parameters:
    movies_df (pd.DataFrame): DataFrame of movies with country and composite score information.

    Returns:
    pd.DataFrame: DataFrame with two columns: 'country' and 'average composite score'.
    """

    avg_score_by_country = (
        movies_df.groupby('country')['composite_score'].mean()
        .reset_index(name='average composite score')
        .sort_values(by='average composite score', ascending=False)
    )

    return avg_score_by_country


def weighted_average_composite_score_by_country(movies_df):
    """
    Calculate the weighted average composite score for each country and return a DataFrame.

    Parameters:
    movies_df (pd.DataFrame): DataFrame of movies with country and composite score information.

    Returns:
    pd.DataFrame: DataFrame with two columns: 'country' and 'weighted average composite score'.
    """

    weighted_avg_score_by_country = (
        movies_df.groupby('country')
        .apply(lambda x: np.average(x['composite_score'], weights=x['numVotes']))
        .reset_index(name='weighted average composite score')
        .sort_values(by='weighted average composite score', ascending=False)
    )

    return weighted_avg_score_by_country


def sort_by_column_and_select(df, sort_column, target_columns):
    """
    Sorts a pandas dataframe by a specified column and selects only desired columns.

    Args:
      df (pandas.DataFrame): The dataframe to be sorted and filtered.
      sort_column (str): The column name to sort by.
      target_columns (list): A list of column names to select after sorting.

    Returns:
      pandas.DataFrame: A new dataframe sorted by the specified column and containing only the desired columns.
    """

    # Sort by the specified column in descending order (highest first)
    df_sorted = df.sort_values(by=sort_column, ascending=False)

    # Select only the desired columns
    df_filtered = df_sorted[target_columns]

    return df_filtered


def filter_countries_with_reference(df, col_name, reference_df, reference_column, year='2023'):
    """ Filters a pandas dataframe to keep only rows with countries present in a reference list.

  Args:
      df (pandas.DataFrame): The dataframe containing countries/regions.
      col_name (str, optional): The name of the column containing countries/regions in df.
      reference_df (pandas.DataFrame): The already loaded pandas dataframe containing the reference list of countries.
      reference_column (str): The name of the column from which to filter the reference list.
      year (str): The year from which to filter the reference list. NOTE: str

  Returns:
      pandas.DataFrame: A new dataframe containing only rows with countries present in the reference list.
  """

    # Get the reference countries as a set
    reference_countries = set(reference_df[reference_column].tolist())

    # Filter the dataframe based on the reference list
    df_filtered = df[df[col_name].isin(reference_countries)]

    # Sort and select columns (assuming sort_by_column_and_select function exists)
    df_filtered = sort_by_column_and_select(df_filtered, year, ['Country Name', year])

    return df_filtered


def get_countries_and_clean_orders(df, merge_df, merge_col_left, merge_col_right, cols_to_keep, how='left'):
    """
  Merges a DataFrame with another DataFrame based on specified columns,
  handles missing values in the merge column, and selects desired columns.

  Args:
      df (pd.DataFrame): The DataFrame to be processed and merged.
      merge_df (pd.DataFrame): The DataFrame to be merged with.
      merge_col_left (str): The column name for merging in the left DataFrame.
      merge_col_right (str): The column name for merging in the right DataFrame.
      cols_to_keep (list): The list of column names to be kept in the merge DataFrame.
      how (str, optional): The type of merge to perform. Defaults to 'left'.

  Returns:
      pd.DataFrame: The processed and merged DataFrame with selected columns.
      list: A list of countries excluded due to missing values in the merge column.
  """

    # Merge the DataFrames
    merged_df = df.merge(merge_df, left_on=merge_col_left, right_on=merge_col_right, how=how)

    # Find rows with missing values in the 'name' column (assuming it's from the merge)
    missing_values = merged_df[merged_df['name'].isna()]

    # Drop rows with missing values in 'name' column
    processed_df = merged_df.dropna(subset=['name'])

    # Select desired columns
    processed_df = processed_df[cols_to_keep]

    return processed_df, missing_values['country'].tolist()


def calculate_gdp_per_population(gdp_df, population_df, year):
    """
  Calculates GDP per population for each country and returns a new DataFrame.

  Args:
      gdp_df (pd.DataFrame): DataFrame containing countries and GDP values.
      population_df (pd.DataFrame): DataFrame containing countries and population values.
      year (str): The year the gdp and population values to be taken from. NOTE: str

  Returns:
      pd.DataFrame: A new DataFrame with countries and GDP per population.
  """

    # Create a dictionary with country as key and gdp/population as value
    gdp_pop_dict = {
        country: gdp_df[gdp_df['Country Name'] == country][year].values[0] /
                 population_df[population_df['Country Name'] == country][year].values[0]
        for country in gdp_df['Country Name'].unique()
    }

    # Convert the dictionary to a DataFrame
    gdp_pop_df = pd.DataFrame.from_dict(gdp_pop_dict, orient='index', columns=['gdp_per_population'])
    gdp_pop_df.reset_index(inplace=True)  # Make 'country' a column
    gdp_pop_df.sort_values(by='gdp_per_population', ascending=False, inplace=True)

    return gdp_pop_df


def rename_and_add_rank(df, new_col_names):
    """
    Rename the columns of a DataFrame and add a 'rank' column based on row order.
    The DataFrame index is reset to ensure a proper sequential order.

    Parameters:
    df (pd.DataFrame): DataFrame to be modified.
    new_col_names (list): List containing the new column names (must contain exactly two names).

    Returns:
    pd.DataFrame: DataFrame with renamed columns and a new 'rank' column.
    """
    if len(df.columns) != 2:
        raise ValueError("The DataFrame must have exactly two columns.")
    if len(new_col_names) != 2:
        raise ValueError("You must provide exactly two new column names.")

    # Rename the columns
    df.columns = new_col_names

    # Reset the index and add the rank column
    df = df.reset_index(drop=True)
    df['rank'] = df.index + 1

    return df


def compute_hegemony(df1, df2, label1, label2):
    """
    Compute and print the hegemony rankings based on the difference in ranks between two DataFrames.

    Args:
    df1 (pd.DataFrame): The first DataFrame containing 'country' and 'rank' columns.
    df2 (pd.DataFrame): The second DataFrame containing 'country' and 'rank' columns.
    label1 (str): The label for the first DataFrame (used in the print title and column suffix).
    label2 (str): The label for the second DataFrame (used in the print title and column suffix).

    Returns:
    pd.DataFrame: A DataFrame with 'country', 'hegemony_score', and 'hegemony_rank' columns.
    """
    # Merge DataFrames on 'country' and calculate hegemony score
    merged_df = df1[['country', 'rank']].merge(df2[['country', 'rank']], on='country',
                                               suffixes=(f'_{label1}', f'_{label2}'))
    merged_df['hegemony_score'] = (merged_df[f'rank_{label1}'] - merged_df[f'rank_{label2}']).abs()

    # Sort by hegemony score and assign hegemony rank
    merged_df = merged_df.sort_values('hegemony_score').reset_index(drop=True)
    merged_df['hegemony_rank'] = merged_df.index + 1

    # Print the results
    print(f"\n{label1.capitalize()} / {label2.capitalize()} Hegemony Rankings:")
    for index, row in merged_df.iterrows():
        print(f"{row['hegemony_rank']}. {row['country']} (Hegemony Score: {row['hegemony_score']})")

    return merged_df[['country', 'hegemony_score', 'hegemony_rank']]
