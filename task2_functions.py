import pandas as pd
import numpy as np


def total_votes_by_country(movies_df):
    """
    Calculate the total sum of votes for each country.

    Parameters:
    movies_df (pd.DataFrame): DataFrame of movies with country and composite score information.

    Returns:
    pd.Series: Series with the total sum of votes for each country.
    """
    votes_by_country = movies_df.groupby('country')['numVotes'].sum()
    votes_by_country.sort_values(ascending=False, inplace=True)

    return votes_by_country


def average_composite_score_by_country(movies_df):
    """
    Calculate the average composite score for each country.

    Parameters:
    movies_df (pd.DataFrame): DataFrame of movies with country and composite score information.

    Returns:
    pd.Series: Series with the average composite score for each country.
    """
    return movies_df.groupby('country')['composite_score'].mean()


def weighted_average_composite_score_by_country(movies_df):
    """
    Calculate the weighted average composite score for each country.

    Parameters:
    movies_df (pd.DataFrame): DataFrame of movies with country and composite score information.

    Returns:
    pd.Series: Series with the weighted average composite score for each country.
    """
    # Calculate the weighted average using the number of votes as weights
    return movies_df.groupby('country').apply(lambda x: np.average(x['composite_score'], weights=x['numVotes']))
