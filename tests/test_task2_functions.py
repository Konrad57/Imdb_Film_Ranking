import unittest
import pandas as pd
from functions.task2_functions import (
    total_votes_by_country,
    average_composite_score_by_country,
    weighted_average_composite_score_by_country,
    sort_by_column_and_select,
    filter_countries_with_reference,
    get_countries_and_clean_orders,
    calculate_gdp_per_population,
    compute_hegemony
)


class TestTask2Functions(unittest.TestCase):

    def setUp(self):
        """Set up mock data for testing."""
        self.movies_df = pd.DataFrame({
            'tconst': ['tt0000001', 'tt0000002', 'tt0000003'],
            'titleType': ['movie', 'movie', 'movie'],
            'startYear': [2000, 2005, 2010],
            'numVotes': [1000, 500, 700],
            'composite_score': [7.5, 8.0, 6.5],
            'country': ['US', 'US', 'FR']
        })

        self.reference_df = pd.DataFrame({
            'country': ['US', 'FR', 'UK'],
            'name': ['United States', 'France', 'United Kingdom']
        })

        self.gdp_df = pd.DataFrame({
            'Country Name': ['US', 'FR', 'UK'],
            '2023': [15000, 20000, 25000]
        })

        self.population_df = pd.DataFrame({
            'Country Name': ['US', 'FR', 'UK'],
            '2023': [300, 400, 500]
        })

    def tearDown(self):
        """Tear down after each test."""
        if hasattr(self, 'errors') and self.errors:
            print("\n".join(self.errors))
            self.errors = []

    def test_total_votes_by_country(self):
        """Test total votes by country."""
        expected_result = pd.DataFrame({
            'country': ['US', 'FR'],
            'number of votes': [1500, 700]
        }).reset_index(drop=True)

        result = total_votes_by_country(self.movies_df)
        pd.testing.assert_frame_equal(result, expected_result)

    def test_average_composite_score_by_country(self):
        """Test average composite score by country."""
        expected_result = pd.DataFrame({
            'country': ['US', 'FR'],
            'average composite score': [7.75, 6.5]
        }).reset_index(drop=True)

        result = average_composite_score_by_country(self.movies_df)
        pd.testing.assert_frame_equal(result, expected_result)

    def test_weighted_average_composite_score_by_country(self):
        """Test weighted average composite score by country."""
        expected_result = pd.DataFrame({
            'country': ['US', 'FR'],
            'weighted average composite score': [7.75, 6.5]
        }).reset_index(drop=True)

        result = weighted_average_composite_score_by_country(self.movies_df)
        pd.testing.assert_frame_equal(result, expected_result)

    def test_sort_by_column_and_select(self):
        """Test sort by column and select."""
        df = pd.DataFrame({
            'col1': ['A', 'B', 'C'],
            'col2': [3, 2, 1]
        })

        expected_result = pd.DataFrame({
            'col1': ['A', 'B', 'C'],
            'col2': [3, 2, 1]
        })

        result = sort_by_column_and_select(df, 'col2', ['col1', 'col2'])
        pd.testing.assert_frame_equal(result, expected_result)

    def test_filter_countries_with_reference(self):
        """Test filtering countries with reference."""
        expected_result = pd.DataFrame({
            'country': ['US', 'FR'],
            'numVotes': [1000, 700]
        }).reset_index(drop=True)

        result = filter_countries_with_reference(self.movies_df, 'country', self.reference_df, 'country')
        pd.testing.assert_frame_equal(result, expected_result)

    def test_get_countries_and_clean_orders(self):
        """Test getting countries and cleaning orders."""
        expected_result = pd.DataFrame({
            'country': ['US', 'FR'],
            'numVotes': [1000, 700]
        }).reset_index(drop=True)

        result, missing_countries = get_countries_and_clean_orders(
            self.movies_df, self.reference_df, 'country', 'country', ['country', 'numVotes']
        )
        pd.testing.assert_frame_equal(result, expected_result)
        self.assertEqual(missing_countries, [])

    def test_calculate_gdp_per_population(self):
        """Test calculating GDP per population."""
        expected_result = pd.DataFrame({
            'country': ['US', 'FR', 'UK'],
            'gdp_per_population': [50, 50, 50]
        }).reset_index(drop=True)

        result = calculate_gdp_per_population(self.gdp_df, self.population_df)
        pd.testing.assert_frame_equal(result, expected_result)

    def test_compute_hegemony(self):
        """Test computing hegemony rankings."""
        df1 = pd.DataFrame({
            'country': ['US', 'FR', 'UK'],
            'rank': [1, 2, 3]
        })

        df2 = pd.DataFrame({
            'country': ['FR', 'US', 'UK'],
            'rank': [1, 2, 3]
        })

        expected_result = pd.DataFrame({
            'country': ['US', 'FR', 'UK'],
            'hegemony_score': [1, 1, 0],
            'hegemony_rank': [1, 2, 3]
        }).reset_index(drop=True)

        result = compute_hegemony(df1, df2, 'label1', 'label2')
        pd.testing.assert_frame_equal(result, expected_result)


if __name__ == '__main__':
    unittest.main()
