import pandas as pd
import os
from task1_functions import quality_of_movies_by_country


def load_data(file_path: str) -> pd.DataFrame:
    """Load a TSV file into a pandas DataFrame."""
    try:
        print(f'Loading data from: {file_path} ...')
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


def save_clean_data(df, output_path):
    """Save cleaned DataFrame to the specified file path."""
    df.to_csv(output_path, index=False)


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
    basics = clean_data(load_data(os.path.join(args.data_dir, 'title.basics.tsv')))
    akas = clean_data(load_data(os.path.join(args.data_dir, 'title.akas.tsv')))
    ratings = clean_data(load_data(os.path.join(args.data_dir, 'title.ratings.tsv')))

    # Analyze the quality of movies by country
    country_counts = quality_of_movies_by_country(basics, ratings, akas, args.top_orders)

    # Save the results
    counts_output_path = os.path.join(args.output_dir, "country_counts.json")
    with open(counts_output_path, 'w') as f:
        json.dump(country_counts, f)

    # Save cleaned data
    save_clean_data(basics, os.path.join(args.output_dir, 'basics.csv'))
    save_clean_data(akas, os.path.join(args.output_dir, 'akas.csv'))
    save_clean_data(ratings, os.path.join(args.output_dir, 'ratings.csv'))
