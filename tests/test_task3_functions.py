import unittest
import pandas as pd
from functions.task3_functions import prepare_movies_directors, rank_directors, custom_ranking


class TestMovieDirectorFunctions(unittest.TestCase):

    def setUp(self):
        """Set up mock data for testing."""
        # Mock data for prepare_movies_directors function
        self.crew_df = pd.DataFrame({
            'directors': ['Director A', 'Director B', 'Director C'],
            'job': ['Director', 'Director', 'Writer'],
            'tconst': ['tt000001', 'tt000002', 'tt000003']
        })

        self.names_df = pd.DataFrame({
            'nconst': ['Director A', 'Director B', 'Director C'],
            'primaryName': ['Director A Name', 'Director B Name', 'Director C Name']
        })

        self.movies_df = pd.DataFrame({
            'tconst': ['tt000001', 'tt000002', 'tt000003', 'tt000004'],
            'title': ['Movie 1', 'Movie 2', 'Movie 3', 'Movie 4']
        })

        # Mock data for rank_directors and custom_ranking functions
        self.movies_scores_df = pd.DataFrame({
            'director': ['Director A', 'Director B', 'Director A', 'Director B'],
            'score': [7.5, 8.0, 6.5, 7.0]
        })

    def tearDown(self):
        """Tear down after each test."""
        if hasattr(self, 'errors') and self.errors:
            print("\n".join(self.errors))
            self.errors = []

    def test_prepare_movies_directors(self):
        """Test preparation of movies with directors."""
        expected_result = pd.DataFrame({
            'directors': ['Director A', 'Director B'],
            'job': ['Director', 'Director'],
            'tconst': ['tt000001', 'tt000002'],
            'primaryName': ['Director A Name', 'Director B Name']
        })

        result = prepare_movies_directors(self.crew_df, self.names_df, self.movies_df)
        pd.testing.assert_frame_equal(result[['directors', 'job', 'tconst', 'primaryName']], expected_result)

    def test_rank_directors_mean_aggregation(self):
        """Test ranking directors with mean aggregation."""
        expected_result = pd.DataFrame({
            'director': ['Director B', 'Director A'],
            'aggregated_score': [7.5, 7.0],
            'total_movies': [2, 2],
            'rank': [1.0, 2.0]
        })

        result = rank_directors(self.movies_scores_df, 'director', 'score', aggregation='mean')
        pd.testing.assert_frame_equal(result, expected_result)

    def test_rank_directors_sum_aggregation(self):
        """Test ranking directors with sum aggregation."""
        expected_result = pd.DataFrame({
            'director': ['Director A', 'Director B'],
            'aggregated_score': [13.5, 15.0],
            'total_movies': [2, 2],
            'rank': [2.0, 1.0]
        })

        result = rank_directors(self.movies_scores_df, 'director', 'score', aggregation='sum')
        pd.testing.assert_frame_equal(result, expected_result)

    def test_custom_ranking(self):
        """Test custom ranking of directors."""
        expected_result = pd.DataFrame({
            'director': ['Director B', 'Director A'],
            'custom_score': [16.0, 7.5],
            'total_movies': [2, 2],
            'rank': [1.0, 2.0]
        })

        result = custom_ranking(self.movies_scores_df, 'director', 'score', good_threshold=8.0, bad_threshold=5.0)
        pd.testing.assert_frame_equal(result, expected_result)

    def test_custom_ranking_with_composite_score(self):
        """Test custom ranking of directors using composite_score."""
        movies_scores_df = pd.DataFrame({
            'director': ['Director B', 'Director A'],
            'composite_score': [8.0, 7.5]
        })

        expected_result = pd.DataFrame({
            'director': ['Director B', 'Director A'],
            'custom_score': [16.0, 7.5],
            'total_movies': [2, 2],
            'rank': [1.0, 2.0]
        })

        result = custom_ranking(movies_scores_df, 'director', 'composite_score', good_threshold=8.0, bad_threshold=5.0)
        pd.testing.assert_frame_equal(result, expected_result)

    def test_custom_ranking_invalid_thresholds(self):
        """Test custom ranking with invalid threshold values."""
        with self.assertRaises(TypeError):
            custom_ranking(self.movies_scores_df, 'director', 'score', good_threshold='invalid', bad_threshold=5.0)


if __name__ == '__main__':
    unittest.main()
