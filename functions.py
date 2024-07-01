import pandas as pd
import os


def load_data(file_path: str) -> pd.DataFrame:
    """Load a TSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path, sep='\t', low_memory=False)
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        raise
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic data cleaning."""
    df.replace({'\\N': pd.NA}, inplace=True)
    return df


def load_and_clean_all_data(data_dir: str) -> dict:
    """Load and clean all relevant IMDb datasets."""
    datasets = {
        'basics': 'title.basics.tsv',
        'akas': 'title.akas.tsv',
        # 'crew': 'title.crew.tsv',
        'ratings': 'title.ratings.tsv',
        # 'names': 'name.basics.tsv'
    }

    data = {}
    for key, filename in datasets.items():
        file_path = os.path.join(data_dir, filename)
        print(f"Loading {file_path}...")
        df = load_data(file_path)
        df = clean_data(df)
        data[key] = df

    return data


def save_clean_data(data: dict, output_dir: str):
    """Save cleaned datasets to the specified directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for key, df in data.items():
        output_path = os.path.join(output_dir, f"{key}.csv")
        df.to_csv(output_path, index=False)


def merge_datasets(dfs, keys):
    """
    Merge a list of datasets on the specified keys.

    Parameters:
    dfs (list): List of DataFrames to merge.
    keys (list): List of keys to merge on. Each element in keys corresponds to a key in the dfs list.

    Returns:
    pd.DataFrame: Merged DataFrame.
    """
    merged_df = dfs[0]
    for i in range(1, len(dfs)):
        merged_df = pd.merge(merged_df, dfs[i], left_on=keys[i - 1][0], right_on=keys[i - 1][1])
    print(f"Merged dataset size: {merged_df.shape}")
    return merged_df


def filter_movies(merged_df):
    """Filter the merged dataset to include only movies."""
    movies_df = merged_df[merged_df['titleType'] == 'movie']
    # movies_df = movies_df[movies_df['isOriginalTitle'] == 1]
    print(f"Filtered dataset size: {movies_df.shape}")
    return movies_df


def get_movie_country(title_akas):
    """
    Establish the country of origin for each movie, ensuring rows with isOriginalTitle = 1 have a region code.

    Parameters:
    title_akas (pd.DataFrame): DataFrame containing title.akas data.

    Returns:
    pd.DataFrame: DataFrame with columns ['titleId', 'country'].
    """
    # Filter to original titles
    original_titles = title_akas[title_akas['isOriginalTitle'] == 1][['titleId', 'region', 'isOriginalTitle']]

    # Find missing regions
    missing_regions = original_titles['region'].isna()

    # Fill missing regions by looking at other rows with the same titleId
    if missing_regions.any():
        # Create a DataFrame with titleId and non-missing region
        non_missing_regions = title_akas[['titleId', 'region']].dropna().drop_duplicates('titleId')

        # Merge to fill missing regions in original titles
        original_titles = original_titles.merge(non_missing_regions, on='titleId', how='left', suffixes=('', '_fill'))

        # Use the filled regions where necessary
        original_titles['region'] = original_titles['region'].combine_first(original_titles['region_fill'])

        # Drop the temporary fill column
        original_titles.drop(columns=['region_fill'], inplace=True)

    # Filter out rows with missing regions
    original_titles = original_titles.dropna(subset=['region'])

    # Rename 'region' column to 'country'
    original_titles.rename(columns={'region': 'country'}, inplace=True)

    return original_titles[['titleId', 'country']]


def calculate_composite_score(movies_df):
    """Calculate the composite score for each movie."""
    movies_df['composite_score'] = (movies_df['averageRating'] * 0.7) + (movies_df['numVotes'] * 0.3)
    return movies_df


def count_country_appearances(top_movies_df, top_orders):
    """Count how many times each country appears in the specified top N sequences."""
    country_counts = {}
    for n in top_orders:
        top_n_movies = top_movies_df.head(n)
        counts = top_n_movies['country'].value_counts().to_dict()
        country_counts[n] = counts
    return country_counts


def quality_of_movies_by_country(data, top_orders):
    """
    Main function to analyze the quality of movies by country.

    Parameters:
    data (dict): Dictionary of DataFrames containing IMDb data.
    top_orders (list): List of top N orders to analyze.

    Returns:
    dict: Dictionary containing counts of country appearances in specified top N sequences.
    """
    # Merge the relevant datasets
    merged_df = merge_datasets([data['basics'], data['ratings'], data['akas']],
                               [('tconst', 'tconst'), ('tconst', 'titleId')])

    # Filter the merged dataset to include only movies
    movies_df = filter_movies(merged_df)

    # Get the country of origin for each movie
    country_df = get_movie_country(data['akas'])
    movies_df = pd.merge(movies_df, country_df, left_on='tconst', right_on='titleId')
    movies_df = movies_df[movies_df['isOriginalTitle'] == 1]
    print(f"Size of movies_df after country assignment: {movies_df.shape}")

    # Calculate the composite score for each movie
    movies_df = calculate_composite_score(movies_df)

    # Sort movies by composite score
    movies_df = movies_df.sort_values(by='composite_score', ascending=False)
    columns_to_display = ['tconst', 'primaryTitle', 'originalTitle', 'averageRating', 'country', 'composite_score']
    print(movies_df[columns_to_display].head())

    # Count country appearances in specified top N sequences
    country_counts = count_country_appearances(movies_df, top_orders)

    return country_counts


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Load and clean IMDb datasets and analyze the quality of movies by country.")
    parser.add_argument("data_dir", type=str, help="Directory containing the raw data files.")
    parser.add_argument("output_dir", type=str, help="Directory to save the cleaned data files and results.")
    parser.add_argument("--top_orders", type=int, nargs='+', default=[10, 20, 100, 200],
                        help="List of top N orders to analyze.")

    args = parser.parse_args()

    # Load and clean the data
    data = load_and_clean_all_data(args.data_dir)

    # Analyze the quality of movies by country
    country_counts = quality_of_movies_by_country(data, args.top_orders)

    # Save the results
    counts_output_path = os.path.join(args.output_dir, "country_counts.json")
    with open(counts_output_path, 'w') as f:
        json.dump(country_counts, f)

    # Save cleaned data
    save_clean_data(data, args.output_dir)
