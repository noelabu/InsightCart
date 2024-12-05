import pandas as pd
from scipy import stats
from swarm import Swarm, Agent
import logging

class DataPreprocessing:
    def __init__(self, cleaning_report, anomaly_report):
        """
        Initialize the data preprocessing swarm with cleaning and anomaly reports
        
        Args:
            cleaning_report (dict): Detailed cleaning recommendations
            anomaly_report (dict): Anomaly detection recommendations
        """
        self.cleaning_report = cleaning_report
        self.anomaly_report = anomaly_report
        self.client = Swarm()
        self._setup_agents()

    def _setup_agents(self):
        """
        Set up specialized agents for different preprocessing tasks
        """
        # Data Type Conversion Agent
        def convert_data_types(dataframe):
            return self._convert_data_types(dataframe)

        self.type_conversion_agent = Agent(
            name="DataTypeAgent",
            model="gpt-4o-mini",
            instructions="Convert data types based on provided recommendations. Handle type conversions carefully, ensuring data integrity.",
            functions=[convert_data_types]
        )

        # Missing Value Handling Agent
        def handle_missing_values(dataframe):
            return self._handle_missing_values(dataframe)

        self.missing_values_agent = Agent(
            name="MissingValuesAgent",
            instructions="Handle missing values using appropriate statistical techniques. Prefer median for normal distributions, mode for skewed distributions.",
            functions=[handle_missing_values]
        )

        # Outlier Handling Agent
        def process_outliers(dataframe):
            return self._process_outliers(dataframe)

        self.outlier_agent = Agent(
            name="OutlierAgent",
            instructions="Detect and handle outliers using Z-score method. Remove or normalize outliers based on pre-defined thresholds.",
            functions=[process_outliers]
        )

        # Anomaly Filtering Agent
        def filter_anomalies(dataframe):
            return self._filter_anomalies(dataframe)

        self.anomaly_agent = Agent(
            name="AnomalyAgent",
            instructions="Identify and filter out data anomalies using quantile-based filtering. Ensure data quality while minimizing information loss.",
            functions=[filter_anomalies]
        )

    def preprocess_data(self, dataframe):
        """
        Orchestrate preprocessing steps using agent-based workflow
        
        Args:
            dataframe (pd.DataFrame): Input dataframe to be preprocessed
        
        Returns:
            pd.DataFrame: Fully preprocessed dataframe
        """
        # Validate input
        if dataframe is None or dataframe.empty:
            logging.warning("Input dataframe is None or empty")
            return pd.DataFrame()

        # Preprocessing pipeline with agent handoffs
        preprocessing_steps = [
            (self.type_conversion_agent, "Convert data types"),
            (self.missing_values_agent, "Handle missing values"),
            (self.outlier_agent, "Process outliers"),
            (self.anomaly_agent, "Filter anomalies")
        ]

        dataframe['revenue'] = dataframe['revenue'].apply(lambda x: int(x.replace("₱", "").replace(",", "")))
        dataframe['ad spend'] = dataframe['revenue'].apply(lambda x: int(x.replace("₱", "").replace(",", "")))
        processed_df = dataframe.copy().to_dict()
        for agent, step_name in preprocessing_steps:
            try:
                response = self.client.run(
                    agent=agent,
                    messages=[{"role": "user", "content": step_name, "dataframe": processed_df}]
                )
                processed_df = response.messages[-1].get("dataframe", processed_df)
            except Exception as e:
                logging.error(f"Error in {step_name}: {e}")
                continue

        return processed_df

    def _convert_data_types(self, dataframe_dict):
        """
        Convert data types based on cleaning recommendations
        
        Returns:
            pd.DataFrame: Dataframe with corrected data types
        """
        df = pd.DataFrame.from_dict(dataframe_dict)
        recommendations = self.cleaning_report.get('recommendations', {}).get('data_type_issues', {})
        
        for column, type_info in recommendations.items():
            if column in df.columns:
                recommended_type = type_info.get('recommended_type')
                try:
                    if recommended_type == 'datetime':
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                    elif recommended_type == 'categorical':
                        df[column] = pd.Categorical(df[column])
                    elif recommended_type == 'boolean':
                        df[column] = df[column].map({'yes': True, 'no': False})
                    elif recommended_type == 'float64':
                        df[column] = pd.to_numeric(df[column].replace('[\$,]', '', regex=True), errors='coerce')
                except Exception as e:
                    logging.error(f"Error converting {column} to {recommended_type}: {e}")
        
        return df

    def _handle_missing_values(self, dataframe_dict):
        """
        Handle missing values using appropriate statistical techniques
        
        
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        df = pd.DataFrame.from_dict(dataframe_dict)
        missing_config = self.cleaning_report.get('recommendations', {}).get('missing_values', {})
        
        if missing_config.get('action') == 'None required':
            return df
        
        for column in df.columns:
            if df[column].isnull().any():
                try:
                    if self._is_normally_distributed(df[column]):
                        df[column].fillna(df[column].median(), inplace=True)
                    else:
                        df[column].fillna(df[column].mode()[0], inplace=True)
                except Exception as e:
                    logging.error(f"Error handling missing values in {column}: {e}")
        
        return df

    def _process_outliers(self, dataframe_dict):
        """
        Process outliers using Z-score method
        
        Returns:
            pd.DataFrame: Dataframe with processed outliers
        """
        df = pd.DataFrame.from_dict(dataframe_dict)
        outlier_config = self.anomaly_report.get('cleaning_recommendations', {})
        
        for column, config in outlier_config.items():
            if column in df.columns and 'outliers' in config:
                try:
                    outliers_info = config['outliers']
                    if outliers_info.get('remove'):
                        threshold = outliers_info['remove'].get('values_above')
                        df = df[df[column] <= threshold]
                except Exception as e:
                    logging.error(f"Error processing outliers in {column}: {e}")
        
        return df

    def _filter_anomalies(self, dataframe):
        """
        Filter anomalies using quantile-based approach
        
        Args:
            dataframe (pd.DataFrame): Input dataframe
        
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        df = dataframe.copy()
        anomaly_config = self.anomaly_report.get('cleaning_recommendations', {})
        
        for column, config in anomaly_config.items():
            if column in df.columns:
                try:
                    # Filter using 1st and 99th percentiles if normalize is true
                    if config.get('outliers', {}).get('normalize', False):
                        upper_bound = df[column].quantile(0.99)
                        lower_bound = df[column].quantile(0.01)
                        df = df[(df[column] <= upper_bound) & (df[column] >= lower_bound)]
                except Exception as e:
                    logging.error(f"Error filtering anomalies in {column}: {e}")
        
        return df

    def _is_normally_distributed(self, series, alpha=0.05):
        """
        Check if a series is normally distributed
        
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
            logging.error(f"Distribution test error: {e}")
            return False
