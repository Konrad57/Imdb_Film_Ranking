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


def rank_directors(movies_df, director_column, score_column, aggregation='mean'):
    """
    Rank directors based on a chosen score and add a column with the total number of movies directed by each.

    Args:
    movies_df (pd.DataFrame): DataFrame containing the director and score columns.
    director_column (str): The column name of the director.
    score_column (str): The column name of the score to rank the directors by.
    aggregation (str): The method to aggregate scores for each director ('mean' or 'sum').

    Returns:
    pd.DataFrame: A DataFrame with 'director', 'aggregated_score', 'rank', and 'total_movies' columns.
    """
    # Group by the director column and aggregate the scores
    if aggregation == 'mean':
        aggregated_scores = movies_df.groupby(director_column)[score_column].agg(['mean', 'size']).reset_index()
    elif aggregation == 'sum':
        aggregated_scores = movies_df.groupby(director_column)[score_column].agg(['sum', 'size']).reset_index()
    else:
        raise ValueError("Aggregation method must be 'mean' or 'sum'")

    # Rename columns for clarity
    aggregated_scores.columns = [director_column, 'aggregated_score', 'total_movies']

    # Rank the directors based on aggregated scores
    aggregated_scores['rank'] = aggregated_scores['aggregated_score'].rank(ascending=False, method='min')

    # Sort by rank
    aggregated_scores = aggregated_scores.sort_values('rank').reset_index(drop=True)

    return aggregated_scores


def custom_ranking(movies_df, director_column, score_column, good_threshold=8.0, bad_threshold=5.0):
    """
    Rank directors based on a custom scoring metric that rewards good movies and penalizes bad movies.
    Perform normalization to range [0, 10] for 'composite_score' before ranking.
    Add a column with the total number of movies directed by each director.

    Args:
    movies_df (pd.DataFrame): DataFrame containing the director and score columns.
    director_column (str): The column name of the director.
    score_column (str): The column name of the score to rank the directors by.
    good_threshold (float): Threshold above which a movie is considered 'good'.
    bad_threshold (float): Threshold below which a movie is considered 'bad'.

    Returns:
    pd.DataFrame: A DataFrame with 'director', 'custom_score', 'rank', and 'total_movies' columns.
    """
    # Perform normalization if score_column is 'composite_score'
    if score_column == 'composite_score':
        # Normalize composite_score to range [0, 10]
        max_score = movies_df['composite_score'].max()
        min_score = movies_df['composite_score'].min()
        movies_df['composite_score'] = ((movies_df['composite_score'] - min_score) / (max_score - min_score)) * 10

    # Calculate custom scores
    movies_df['custom_score'] = movies_df[score_column].apply(
        lambda x: abs(x) - bad_threshold if abs(x) <= bad_threshold
        else (abs(x) - bad_threshold) * 2 if abs(x) <= good_threshold
        else (abs(x) - bad_threshold) * 3)

    # Count total number of movies for each director
    movies_df['total_movies'] = movies_df.groupby(director_column)['tconst'].transform('count')

    # Group by the director column and aggregate the custom scores
    aggregated_scores = movies_df.groupby(director_column).agg({
        'custom_score': 'sum',
        'total_movies': 'first'  # Take the first value since it's the same for all rows of the group
    }).reset_index()

    # Rank the directors based on custom scores
    aggregated_scores['rank'] = aggregated_scores['custom_score'].rank(ascending=False, method='min')

    # Sort by rank
    aggregated_scores = aggregated_scores.sort_values('rank').reset_index(drop=True)

    return aggregated_scores