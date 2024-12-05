from swarm import Swarm, Agent
import pandas as pd
import numpy as np
import scipy.stats as stats

class DataDetectiveAgent:
    def __init__(self):
        """
        Initialize the Data Detective Agent
        """
        # Initialize Swarm client
        self.client = Swarm()
        
        self.cleaning_agent = Agent(
            name="Data Cleaning Agent",
            model="gpt-4o-mini",
            instructions=(
                "You are an expert in data cleaning and preprocessing. "
                "Identify and resolve data quality issues with precision. "
                "Provide detailed, actionable recommendations for data cleaning."
                "Respond ONLY with a JSON format output directly addressing the cleaning recommendations. No json in the start of the string."
            )
        )

        self.anomaly_agent = Agent(
            name="Anomaly Detection Agent",
            model="gpt-4o-mini",
            instructions=(
                "You are a specialist in statistical anomaly detection. "
                "Analyze data for unusual patterns, outliers, and potential anomalies. "
                "Provide comprehensive statistical insights."
                "Respond ONLY with a JSON format output directly addressing the cleaning recommendations. No json in the start of the string."
                
            )
        )

        self.transaction_agent = Agent(
            name="Transaction Risk Agent",
            model="gpt-4o-mini",
            instructions=(
                "You are an expert in transaction risk assessment. "
                "Analyze transactions for potential fraud or suspicious activity. "
                "Provide detailed risk scoring and recommendations."
                "Respond ONLY with a JSON format output directly addressing the risk assessment. No json in the start of the string."
            )
        )

    def clean_and_validate_data(self, df: pd.DataFrame) -> dict:
        """
        Perform data cleaning using the Swarm cleaning agent
        
        :param df: Input pandas DataFrame
        :return: Cleaning report and cleaned DataFrame
        """
        # Prepare data issues for analysis
        data_issues = {
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_type_issues': df.dtypes.astype(str).to_dict()
        }

        # Run cleaning agent analysis
        cleaning_response = self.client.run(
            agent=self.cleaning_agent,
            messages=[{
                "role": "user", 
                "content": f"""Analyze these data issues:
                Missing Values: {str(data_issues['missing_values'])}
                Duplicate Rows: {data_issues['duplicate_rows']}
                Data Type Issues: {str(data_issues['data_type_issues'])}
                
                Provide a comprehensive cleaning strategy."""
            }]
        )

        # Perform basic cleaning
        cleaned_df = df.copy()
        cleaned_df.dropna(inplace=True)
        cleaned_df.drop_duplicates(inplace=True)

        return {
            'cleaning_report': cleaning_response.messages[-1]['content'],
            'cleaned_dataframe': cleaned_df,
            'data_issues': data_issues
        }

    def detect_anomalies(self, df: pd.DataFrame) -> dict:
        """
        Perform anomaly detection using the Swarm anomaly agent
        
        :param df: Input pandas DataFrame
        :return: Anomaly detection report
        """
        # Numerical columns for anomaly detection
        numerical_columns = df.select_dtypes(include=[np.number]).columns

        # Statistical summary
        statistical_summary = df[numerical_columns].describe()

        # Anomaly detection using Z-score
        potential_anomalies = {}
        for column in numerical_columns:
            z_scores = np.abs(stats.zscore(df[column]))
            anomalies = df[z_scores > 3]
            potential_anomalies[column] = {
                'anomaly_count': len(anomalies),
                'anomaly_percentage': len(anomalies) / len(df) * 100
            }

        # Run anomaly detection agent
        anomaly_response = self.client.run(
            agent=self.anomaly_agent,
            messages=[{
                "role": "user", 
                "content": f"""Analyze these potential anomalies:
                Statistical Summary: {str(statistical_summary)}
                Potential Anomalies: {str(potential_anomalies)}
                
                Provide a detailed anomaly detection report."""
            }]
        )

        return {
            'anomaly_report': anomaly_response.messages[-1]['content'],
            'potential_anomalies': potential_anomalies,
            'statistical_summary': statistical_summary
        }

    def flag_suspicious_transactions(self, df: pd.DataFrame) -> dict:
        """
        Flag suspicious transactions using the transaction risk agent
        
        :param df: Input pandas DataFrame
        :return: Transaction flagging report
        """
        sus_df = df.copy(deep=True)
        
        # Convert 'revenue' column to numeric
        sus_df["revenue"] = sus_df["revenue"].apply(lambda x: int(str(x).replace("₱", "").replace(",", "")))

        # Define suspicious transaction indicators
        suspicious_indicators = {
            'high_value_transactions': sus_df[sus_df['revenue'] > sus_df['revenue'].quantile(0.95)]
        }

        # Transaction summary
        transaction_summary = sus_df[['revenue', 'transactions', 'device_type']].describe()

        # Run transaction risk agent
        transaction_response = self.client.run(
            agent=self.transaction_agent,
            messages=[{
                "role": "user", 
                "content": f"""Analyze these transaction details:
                Transaction Summary: {transaction_summary}
                Suspicious Indicators: {suspicious_indicators}
                
                Provide a comprehensive transaction risk assessment."""
            }]
        )

        return {
            'transaction_report': transaction_response.messages[-1]['content'],
            'suspicious_indicators': suspicious_indicators
        }

    def comprehensive_analysis(self, df: pd.DataFrame) -> dict:
        """
        Perform a comprehensive multi-agent analysis
        
        :param df: Input pandas DataFrame
        :return: Comprehensive analysis report
        """

        # Perform individual analysis steps
        cleaning_results = self.clean_and_validate_data(df)
        anomaly_results = self.detect_anomalies(cleaning_results['cleaned_dataframe'])
        transaction_results = self.flag_suspicious_transactions(cleaning_results['cleaned_dataframe'])

        # Combine results
        comprehensive_report = {
            'data_cleaning': cleaning_results,
            'anomaly_detection': anomaly_results,
            'transaction_analysis': transaction_results
        }

        return comprehensive_report