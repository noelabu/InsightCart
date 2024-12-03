import pandas as pd
import numpy as np
from scipy import stats

class DataPreprocessingAgent:
    def __init__(self, cleaning_report, anomaly_report):
        """
        Initialize the preprocessing agent with cleaning and anomaly reports

        Args:
            cleaning_report (dict): Detailed cleaning recommendations
            anomaly_report (dict): Anomaly detection recommendations
        """
        self.cleaning_report = cleaning_report
        self.anomaly_report = anomaly_report

    def preprocess_data(self, dataframe):
        """
        Apply comprehensive data preprocessing with enhanced error handling

        Args:
            dataframe (pd.DataFrame): Input dataframe to be preprocessed

        Returns:
            pd.DataFrame: Cleaned and preprocessed dataframe
        """
        # Validate input
        if dataframe is None:
            print("Input dataframe is None")
            return pd.DataFrame()

        if dataframe.empty:
            print("Input dataframe is empty")
            return dataframe

        # Reset index to ensure clean processing
        df = dataframe.reset_index(drop=True)

        # 1. Correct Data Types
        data_type_corrections = self.cleaning_report['cleaning_recommendations']['data_types']['columns']
        for column, dtype in data_type_corrections.items():
            if column in df.columns:
                try:
                    # Convert revenue and ad spend to int
                    if column in ['revenue', 'ad spend']:
                      df[column] = df[column].apply(lambda x: int(x.replace("₱", "").replace(",", "")))

                    if dtype == 'datetime':
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                    elif dtype == 'float':
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                except Exception as e:
                    print(f"Error converting {column} to {dtype}: {e}")

        # 2. Handle Missing Values
        missing_value_config = self.cleaning_report['cleaning_recommendations']['missing_values']
        for column in missing_value_config['columns']:
            if column in df.columns:
                try:
                    if self._is_normally_distributed(df[column]):
                        df[column].fillna(df[column].median(), inplace=True)
                    else:
                        df[column].fillna(df[column].mode()[0], inplace=True)
                except Exception as e:
                    print(f"Error handling missing values in {column}: {e}")

        # 3. Standardize Inconsistent Values
        inconsistent_values = self.cleaning_report['cleaning_recommendations'].get('inconsistent_values', {})  # Get inconsistent_values, handle if missing
        for column_config in inconsistent_values.get('columns', []):  # Iterate through columns, handle if missing
            if isinstance(column_config, dict):  # Check if it's a dictionary as expected
                column = column_config.get('column')  # Extract column name
                config = column_config.get('config', {})  # Extract config, default to empty dict
                if column in df.columns:
                    try:
                        df[column] = df[column].map(config.get('mapping', {})).fillna(df[column])  # Apply mapping if present
                    except Exception as e:
                        print(f"Error standardizing values in {column}: {e}")

        # 4. Handle Outliers using Z-score method
        outlier_columns = self.cleaning_report['cleaning_recommendations']['outliers']['columns']
        for column in outlier_columns:
            if column in df.columns:
                try:
                    df = self._handle_outliers(df, column)
                except Exception as e:
                    print(f"Error handling outliers in {column}: {e}")

        # 5. Anomaly Detection and Filtering
        df = self._filter_anomalies(df)

        return df


    def _is_normally_distributed(self, series, alpha=0.05):
        """
        Check if a series is normally distributed using Shapiro-Wilk test

        Args:
            series (pd.Series): Input series to test
            alpha (float): Significance level for the test

        Returns:
            bool: True if normally distributed, False otherwise
        """
        try:
            cleaned_series = series.dropna()
            if len(cleaned_series) < 3:
                return False

            _, p_value = stats.shapiro(cleaned_series)
            return p_value > alpha
        except Exception as e:
            print(f"Error in distribution test: {e}")
            return False

    def _handle_outliers(self, df, column, threshold=3):
        """
        Handle outliers using Z-score method with additional error handling

        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Column to handle outliers
            threshold (float): Z-score threshold for outlier detection

        Returns:
            pd.DataFrame: Dataframe with outliers removed or replaced
        """
        try:
            z_scores = np.abs(stats.zscore(df[column]))
            return df[z_scores < threshold].copy()
        except Exception as e:
            print(f"Error handling outliers in {column}: {e}")
            return df

    def _filter_anomalies(self, df):
        """
        Filter out anomalies based on the anomaly report with error handling

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Filtered dataframe
        """
        try:
          for column, anomaly_info in self.anomaly_report['cleaning_recommendations'].items():
              if column in df.columns:
                  # Ensure anomaly_count exists and is a valid number
                  anomaly_count = anomaly_info.get('anomaly_count', 0)

                  if isinstance(anomaly_count, (int, float)) and anomaly_count > 1000:
                      # If anomaly_count is more than 1000, filter anomalies
                      upper_bound = df[column].quantile(0.99)
                      lower_bound = df[column].quantile(0.01)
                      df = df[(df[column] <= upper_bound) & (df[column] >= lower_bound)]

          return df
        except Exception as e:
            print(f"Error filtering anomalies: {e}")
            return df

    def generate_preprocessing_report(self, original_df, processed_df):
        """
        Generate a report on preprocessing changes

        Args:
            original_df (pd.DataFrame): Original dataframe
            processed_df (pd.DataFrame): Processed dataframe

        Returns:
            dict: Preprocessing report
        """
        try:
            report = {
                'original_shape': original_df.shape,
                'processed_shape': processed_df.shape,
                'rows_removed': original_df.shape[0] - processed_df.shape[0],
                'columns_processed': list(self.cleaning_report['cleaning_recommendations']['data_types']['columns'].keys())
            }
            return report
        except Exception as e:
            print(f"Error generating preprocessing report: {e}")
            return {}