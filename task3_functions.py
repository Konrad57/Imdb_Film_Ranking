import pandas as pd


def prepare_movies_directors(crew_df, names_df, movies_df):
    """ Merges crew, names, and movies DataFrames, keeping only movies with directors.

  Args:
      crew_df (pandas.DataFrame): The DataFrame containing crew information (including directors).
      names_df (pandas.DataFrame): The DataFrame containing name information.
      movies_df (pandas.DataFrame): The DataFrame containing movie information.

  Returns:
      pandas.DataFrame: The merged DataFrame containing movies with directors and their information.
  """

    # Merge crew and names on directors and nconst
    merged_df = pd.merge(left=crew_df, right=names_df, left_on='directors', right_on='nconst')

    # Further merge with movies on tconst, keeping all movie rows (how='right')
    merged_df = pd.merge(left=merged_df, right=movies_df, left_on='tconst', right_on='tconst', how='right')

    # Drop rows with missing directors
    merged_df = merged_df.dropna(subset=['directors'])

    return merged_df
