import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Movie Analysis Project")
    parser.add_argument('--movie_data_dir', default="C:\\Users\\konra\\PycharmProjects\\Imdb_Film_Ranking\\data_imdb",
                        help='Path to directory with IMDb movie data')
    parser.add_argument('--gdp_pop_data_dir',
                        default="C:\\Users\\konra\\PycharmProjects\\Imdb_Film_Ranking\\data_gdp_population",
                        help='Path to directory with GDP and Population data')
    parser.add_argument('--start_year', type=int, help='Start year for the analysis period')
    parser.add_argument('--end_year', type=int, help='End year for the analysis period')

    args = parser.parse_args()

    # Set environment variables
    os.environ['MOVIE_DATA_PATH'] = args.movie_data_dir
    os.environ['GDP_POP_DATA_PATH'] = args.gdp_pop_data_dir
    if args.start_year:
        os.environ['START_YEAR'] = str(args.start_year)
    if args.end_year:
        os.environ['END_YEAR'] = str(args.end_year)

    # Print environment variables for debugging
    print("MOVIE_DATA_PATH:", os.environ['MOVIE_DATA_PATH'])
    print("GDP_POP_DATA_PATH:", os.environ['GDP_POP_DATA_PATH'])

    print("Launching Jupyter Notebook and executing cells...")

    # Execute Jupyter Notebook
    subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace', 'analysis.ipynb'])

    print("Notebook execution complete. Opening in browser...")

    # Open the notebook after execution
    subprocess.run(['jupyter', 'notebook', 'analysis.ipynb'])


if __name__ == "__main__":
    main()
