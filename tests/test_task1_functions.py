import unittest
import pandas as pd
import numpy as np
from functions.task1_functions import (
    merge_datasets,
    filter_movies,
    calculate_composite_score,
    get_movie_country,
    count_country_appearances,
    quality_of_movies_by_country
)


class TestTask1Functions(unittest.TestCase):

    def setUp(self):
        """Set up mock data for testing."""
        self.errors = []  # List to accumulate error messages

        self.basics = pd.DataFrame({
            'tconst': ['tt0000001', 'tt0000002', 'tt0000003'],
            'titleType': ['short', 'movie', 'movie'],
            'primaryTitle': ['Carmencita', 'Le clown et ses chiens', 'Pauvre Pierrot'],
            'originalTitle': ['Carmencita', 'Le clown et ses chiens', 'Pauvre Pierrot'],
            'isAdult': [0, 0, 0],
            'startYear': [1894, 1892, 1892],
            'endYear': [np.nan, np.nan, np.nan],
            'runtimeMinutes': [1, 5, 4],
            'genres': ['Documentary,Short', 'Animation,Short', 'Animation,Comedy,Romance']
        })

        self.ratings = pd.DataFrame({
            'tconst': ['tt0000001', 'tt0000002', 'tt0000003'],
            'averageRating': [5.6, 6.1, 6.5],
            'numVotes': [1600, 1100, 1200]
        })

        self.akas = pd.DataFrame({
            'titleId': ['tt0000001', 'tt0000002', 'tt0000003'],
            'ordering': [1, 1, 1],
            'title': ['Carmencita', 'Le clown et ses chiens', 'Pauvre Pierrot'],
            'region': ['US', 'FR', 'FR'],
            'language': [np.nan, np.nan, np.nan],
            'types': [np.nan, np.nan, np.nan],
            'attributes': [np.nan, np.nan, np.nan],
            'isOriginalTitle': [1, 1, 1]
        })

    def tearDown(self):
        """Print accumulated error messages after each test."""
        if self.errors:
            print("\n".join(self.errors))
            self.errors = []  # Reset errors after printing

    def test_merge_datasets(self):
        """Test merging of basics, ratings, and akas datasets."""
        try:
            merged_df = merge_datasets(self.basics, self.ratings, self.akas)
            self.assertEqual(merged_df.shape[0], 3)
            self.assertIn('averageRating', merged_df.columns)
            self.assertIn('title', merged_df.columns)
        except AssertionError as e:
            self.errors.append(f"Error in test_merge_datasets: {str(e)}")

    def test_filter_movies(self):
        """Test filtering merged dataset to include only movies."""
        try:
            merged_df = merge_datasets(self.basics, self.ratings, self.akas)
            movies_df = filter_movies(merged_df)
            self.assertEqual(movies_df.shape[0], 2)
            self.assertTrue(all(movies_df['titleType'] == 'movie'))
        except AssertionError as e:
            self.errors.append(f"Error in test_filter_movies: {str(e)}")

    def test_calculate_composite_score(self):
        """Test calculation of composite score for each movie."""
        try:
            merged_df = merge_datasets(self.basics, self.ratings, self.akas)
            movies_df = filter_movies(merged_df)
            movies_df = calculate_composite_score(movies_df)
            self.assertIn('composite_score', movies_df.columns)
            expected_scores = (movies_df['averageRating'] * 0.7) + (movies_df['numVotes'] * 0.3)
            pd.testing.assert_series_equal(movies_df['composite_score'], expected_scores, check_names=False)
        except AssertionError as e:
            self.errors.append(f"Error in test_calculate_composite_score: {str(e)}")

    def test_get_movie_country(self):
        """Test establishing the country of origin for each movie."""
        try:
            merged_df = merge_datasets(self.basics, self.ratings, self.akas)
            movies_df = filter_movies(merged_df)
            country_df = get_movie_country(movies_df)
            self.assertIn('country', country_df.columns)
            self.assertEqual(country_df.shape[0], 2)
        except AssertionError as e:
            self.errors.append(f"Error in test_get_movie_country: {str(e)}")

    def test_count_country_appearances(self):
        """Test counting how many times each country appears in the specified top N sequences."""
        try:
            merged_df = merge_datasets(self.basics, self.ratings, self.akas)
            movies_df = filter_movies(merged_df)
            movies_df = calculate_composite_score(movies_df)
            country_df = get_movie_country(movies_df)
            movies_df = pd.merge(movies_df, country_df, left_on='tconst', right_on='titleId')
            movies_df = movies_df.sort_values(by='composite_score', ascending=False)
            top_orders = [1, 2]
            country_counts = count_country_appearances(movies_df, top_orders)
            self.assertIn(1, country_counts)
            self.assertIn(2, country_counts)
            self.assertEqual(country_counts[1], {'FR': 1})
            self.assertEqual(country_counts[2], {'FR': 2})
        except AssertionError as e:
            self.errors.append(f"Error in test_count_country_appearances: {str(e)}")

    def test_quality_of_movies_by_country(self):
        """Test main function to analyze the quality of movies by country."""
        try:
            top_orders = [1, 2]
            country_counts, movies_df = quality_of_movies_by_country(self.basics, self.ratings, self.akas, top_orders)
            self.assertIn(1, country_counts)
            self.assertIn(2, country_counts)
            self.assertIn('composite_score', movies_df.columns)
            self.assertIn('country', movies_df.columns)
        except AssertionError as e:
            self.errors.append(f"Error in test_quality_of_movies_by_country: {str(e)}")


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
