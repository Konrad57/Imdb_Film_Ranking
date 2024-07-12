import unittest
from unittest.mock import patch, mock_open
import pandas as pd
from functions.utilities import load_data, clean_data, save_clean_data


class TestUtilities(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data="tconst\ttitleType\tprimaryTitle\t...\n")
    @patch("os.path.splitext", return_value=('.tsv', '.tsv'))
    @patch("pandas.read_csv")
    def test_load_data_tsv(self, mock_read_csv, mock_splitext, mock_open):
        """Test loading a TSV file into a pandas DataFrame."""
        mock_df = pd.DataFrame({'tconst': ['tt0000001', 'tt0000002'], 'titleType': ['movie', 'short']})
        mock_read_csv.return_value = mock_df
        file_path = 'test.tsv'
        df = load_data(file_path)
        mock_read_csv.assert_called_once_with(file_path, sep='\t', low_memory=False)
        self.assertTrue(mock_open.called)
        self.assertIsInstance(df, pd.DataFrame)

    @patch("builtins.open", new_callable=mock_open, read_data="tconst,titleType,primaryTitle,...\n")
    @patch("os.path.splitext", return_value=('.csv', '.csv'))
    @patch("pandas.read_csv")
    def test_load_data_csv(self, mock_read_csv, mock_splitext, mock_open):
        """Test loading a CSV file into a pandas DataFrame."""
        mock_df = pd.DataFrame({'tconst': ['tt0000001', 'tt0000002'], 'titleType': ['movie', 'short']})
        mock_read_csv.return_value = mock_df
        file_path = 'test.csv'
        df = load_data(file_path)
        mock_read_csv.assert_called_once_with(file_path, sep=',', low_memory=False, header=0)
        self.assertTrue(mock_open.called)
        self.assertIsInstance(df, pd.DataFrame)

    @patch("pandas.read_csv", side_effect=FileNotFoundError)
    def test_load_data_file_not_found(self, mock_read_csv):
        """Test handling of file not found error."""
        with self.assertRaises(FileNotFoundError):
            load_data('non_existent_file.csv')

    @patch("builtins.print")
    def test_clean_data_inconsistencies(self, mock_print):
        """Test cleaning the data and detecting inconsistencies."""
        df = pd.DataFrame({
            'tconst': ['tt0000001', 'tt0000002', None],
            'titleType': ['movie', 'short', 'movie'],
            'primaryTitle': ['Title1', None, 'Title3']
        })
        cleaned_df = clean_data(df)
        expected_df = pd.DataFrame({
            'tconst': ['tt0000001', 'tt0000002', pd.NA],
            'titleType': ['movie', 'short', 'movie'],
            'primaryTitle': ['Title1', pd.NA, 'Title3']
        })
        pd.testing.assert_frame_equal(cleaned_df, expected_df)
        mock_print.assert_any_call("Warning: 1 missing values detected in 'tconst' column.")
        mock_print.assert_any_call("Warning: 1 missing values detected in 'primaryTitle' column.")

    @patch("pandas.DataFrame.to_csv")
    def test_save_clean_data(self, mock_to_csv):
        """Test saving cleaned DataFrame to the specified file path."""
        df = pd.DataFrame({'col1': ['value1', 'value2'], 'col2': ['value3', 'value4']})
        output_path = 'output.csv'
        save_clean_data(df, output_path)
        mock_to_csv.assert_called_once_with(output_path, index=False)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
