import pandas as pd


def merge_datasets(basics, ratings, akas):
    """
    Merge the basics, ratings, and akas datasets.

    Parameters:
    basics (pd.DataFrame): DataFrame containing basics data.
    ratings (pd.DataFrame): DataFrame containing ratings data.
    akas (pd.DataFrame): DataFrame containing akas data.

    Returns:
    pd.DataFrame: Merged DataFrame.
    """
    merged_df = basics.merge(ratings, on='tconst', how='left').merge(akas, left_on='tconst', right_on='titleId',
                                                                     how='left')
    return merged_df


def filter_movies(merged_df):
    """Filter the merged dataset to include only movies."""
    movies_df = merged_df[merged_df['titleType'] == 'movie']
    return movies_df


def prepare_data(basics, ratings, akas):

    merged_df = merge_datasets(basics, ratings, akas)
    movies_df = filter_movies(merged_df)

    return movies_df


def calculate_composite_score(movies_df):
    """Calculate the composite score for each movie."""
    if 'averageRating' not in movies_df.columns or 'numVotes' not in movies_df.columns:
        print("Warning: Missing 'averageRating' or 'numVotes' columns.")
    else:
        movies_df['composite_score'] = (movies_df['averageRating'] * 0.7) + (movies_df['numVotes'] * 0.3)
    return movies_df


def get_movie_country(movie_df):
    """
    Establish the country of origin for each movie, ensuring rows with isOriginalTitle = 1 have a region code.

    Parameters:
    movie_df (pd.DataFrame): DataFrame containing the merged and filtered movie data.

    Returns:
    pd.DataFrame: DataFrame with columns ['titleId', 'country'].
    """
    original_titles = movie_df[movie_df['isOriginalTitle'] == 1][['titleId', 'title', 'region', 'isOriginalTitle']]

    missing_regions = original_titles['region'].isna()
    if missing_regions.any():
        non_missing_regions = movie_df[['titleId', 'title', 'region']].dropna().drop_duplicates(['titleId', 'title'])
        original_titles = original_titles.merge(
            non_missing_regions,
            on=['titleId', 'title'],
            how='left',
            suffixes=('', '_fill')
        )
        original_titles['region'] = original_titles['region'].combine_first(original_titles['region_fill'])
        original_titles.drop(columns=['region_fill'], inplace=True)

    remaining_missing = original_titles['region'].isna()
    if remaining_missing.any():
        additional_filled_regions = movie_df[['titleId', 'region']].dropna().drop_duplicates('titleId')
        original_titles = original_titles.merge(
            additional_filled_regions,
            on='titleId',
            how='left',
            suffixes=('', '_additional')
        )
        original_titles['region'] = original_titles['region'].combine_first(original_titles['region_additional'])
        original_titles.drop(columns=['region_additional'], inplace=True)

    num_missing_country = original_titles['region'].isna().sum()
    if num_missing_country > 0:
        print(f"There are {num_missing_country} movies without an assigned country.")

    original_titles = original_titles.dropna(subset=['region'])
    original_titles.rename(columns={'region': 'country'}, inplace=True)

    return original_titles[['titleId', 'country']]


def count_country_appearances(top_movies_df, top_orders):
    """Count how many times each country appears in the specified top N sequences."""
    country_counts = {}
    for n in top_orders:
        top_n_movies = top_movies_df.head(n)
        counts = top_n_movies['country'].value_counts().to_dict()
        country_counts[n] = counts
    return country_counts


def quality_of_movies_by_country(movies_df, top_orders):
    """
    Main function to analyze the quality of movies by country.

    Parameters:
    basics (pd.DataFrame): DataFrame containing basics data.
    ratings (pd.DataFrame): DataFrame containing ratings data.
    akas (pd.DataFrame): DataFrame containing akas data.
    top_orders (list): List of top N orders to analyze.

    Returns:
    tuple: A tuple containing:
        - country_counts (dict): Dictionary containing counts of country appearances in specified top N sequences.
        - movies_df (pd.DataFrame): DataFrame of movies with additional country and composite score information.
    """

    country_df = get_movie_country(movies_df)
    movies_df = pd.merge(movies_df, country_df, left_on='tconst', right_on='titleId')
    movies_df = movies_df[movies_df['isOriginalTitle'] == 1]

    movies_df = calculate_composite_score(movies_df)
    movies_df = movies_df.sort_values(by='composite_score', ascending=False)

    country_counts = count_country_appearances(movies_df, top_orders)

    return country_counts, movies_df
