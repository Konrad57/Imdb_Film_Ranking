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
        'crew': 'title.crew.tsv',
        'ratings': 'title.ratings.tsv',
        'names': 'name.basics.tsv'
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
    return merged_df


def filter_movies(merged_df):
    """Filter the merged dataset to include only movies."""
    movies_df = merged_df[merged_df['titleType'] == 'movie']
    return movies_df


def get_top_n_movies(movies_df, top_n):
    """Select the top N movies based on average rating."""
    top_movies_df = movies_df.sort_values(by='averageRating', ascending=False).head(top_n)
    return top_movies_df


def calculate_country_quality(top_movies_df):
    """Calculate the quality scores by country."""
    country_quality = top_movies_df.groupby('region').agg(
        avg_rating=('averageRating', 'mean'),
        total_votes=('numVotes', 'sum')
    ).reset_index()
    country_quality['composite_score'] = (country_quality['avg_rating'] * 0.7) + (country_quality['total_votes'] * 0.3)
    country_quality = country_quality.sort_values(by='composite_score', ascending=False)
    return country_quality


def quality_of_movies_by_country(data, top_n):
    """
    Main function to analyze the quality of movies by country.

    Parameters:
    data (dict): Dictionary of DataFrames containing IMDb data.
    top_n (int): Number of top movies to analyze.

    Returns:
    pd.DataFrame: DataFrame containing the quality scores by country.
    """
    # Merge the relevant datasets
    merged_df = merge_datasets([data['basics'], data['ratings'], data['akas']],
                               [('tconst', 'tconst'), ('tconst', 'titleId')])

    # Filter the merged dataset to include only movies
    movies_df = filter_movies(merged_df)

    # Select the top N movies
    top_movies_df = get_top_n_movies(movies_df, top_n)

    # Calculate the quality scores by country
    country_quality = calculate_country_quality(top_movies_df)

    return country_quality


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and clean IMDb datasets and analyze the quality of movies by country.")
    parser.add_argument("data_dir", type=str, help="Directory containing the raw data files.")
    parser.add_argument("output_dir", type=str, help="Directory to save the cleaned data files and results.")
    parser.add_argument("--top_n", type=int, default=200, help="Number of top movies to analyze.")

    args = parser.parse_args()

    # Analyze the quality of movies by country
    country_quality = quality_of_movies_by_country(args.data_dir, args.top_n)

    # Save the results
    output_path = os.path.join(args.output_dir, "country_quality.csv")
    country_quality.to_csv(output_path, index=False)

    # Save cleaned data
    data = load_and_clean_all_data(args.data_dir)
    save_clean_data(data, args.output_dir)
