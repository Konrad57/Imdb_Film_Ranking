import pandas as pd
import numpy as np


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
    merged_df = basics.merge(ratings, on='tconst').merge(akas, left_on='tconst', right_on='titleId')
    return merged_df


def filter_movies(merged_df):
    """Filter the merged dataset to include only movies."""
    movies_df = merged_df[merged_df['titleType'] == 'movie']
    return movies_df


def calculate_composite_score(movies_df):
    """Calculate the composite score for each movie."""
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
    # Filter to original titles
    original_titles = movie_df[movie_df['isOriginalTitle'] == 1][['titleId', 'title', 'region', 'isOriginalTitle']]

    # Find missing regions
    missing_regions = original_titles['region'].isna()

    # Fill missing regions by looking at other rows with the same titleId and matching title
    if missing_regions.any():
        # Create a DataFrame with titleId, title, and non-missing region
        non_missing_regions = movie_df[['titleId', 'title', 'region']].dropna().drop_duplicates(['titleId', 'title'])

        # Merge to fill missing regions in original titles based on titleId and title
        original_titles = original_titles.merge(
            non_missing_regions,
            on=['titleId', 'title'],
            how='left',
            suffixes=('', '_fill')
        )

        # Use the filled regions where necessary
        original_titles['region'] = original_titles['region'].combine_first(original_titles['region_fill'])

        # Drop the temporary fill column
        original_titles.drop(columns=['region_fill'], inplace=True)

    # If some regions are still missing, fill them with any available non-missing region with the same titleId
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

    # Print the number of movies without an assigned country before filtering
    num_missing_country = original_titles['region'].isna().sum()
    if num_missing_country > 0:
        print(f"There are {num_missing_country} movies without an assigned country.")

    # Filter out rows with missing regions
    original_titles = original_titles.dropna(subset=['region'])

    # Rename 'region' column to 'country'
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


def quality_of_movies_by_country(basics, ratings, akas, top_orders):
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
    # Merge the relevant datasets
    merged_df = merge_datasets(basics, ratings, akas)

    # Filter the merged dataset to include only movies
    movies_df = filter_movies(merged_df)

    # Get the country of origin for each movie
    country_df = get_movie_country(movies_df)
    movies_df = pd.merge(movies_df, country_df, left_on='tconst', right_on='titleId')
    movies_df = movies_df[movies_df['isOriginalTitle'] == 1]

    # Calculate the composite score for each movie
    movies_df = calculate_composite_score(movies_df)

    # Sort movies by composite score
    movies_df = movies_df.sort_values(by='composite_score', ascending=False)

    # Count country appearances in specified top N sequences
    country_counts = count_country_appearances(movies_df, top_orders)

    return country_counts, movies_df
