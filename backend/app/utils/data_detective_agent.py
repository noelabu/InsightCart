import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Dict, Any
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class DataDetectiveLLMAgent:
    def __init__(self, openai_api_key: str):
        """
        Initialize the Data Detective LLM Agent

        :param openai_api_key: OpenAI API key for authentication
        """
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=2500,
            api_key=openai_api_key
        )

        # Data Cleaning Prompt
        self.data_cleaning_prompt = PromptTemplate(
            input_variables=['data_issues', 'column_summary'],
            template="""
            Data Cleaning and Validation Report

            Identified Data Issues:
            {data_issues}

            Column Summary:
            {column_summary}

            Comprehensive Data Cleaning Strategy:
            1. Analyze and categorize data quality issues
            2. Recommend precise cleaning techniques
            3. Suggest data transformation approaches
            4. Provide risk assessment for each cleaning method

            Cleaning Recommendations Should:
            - Preserve data integrity
            - Minimize information loss
            - Provide transparent reasoning
            - Offer multiple cleaning approaches

            Explanation of Fields:
            - **missing_values**: Columns with missing data and the method for filling those gaps (e.g., "mean", "median").
            - **duplicates**: Columns where duplicates need to be removed.
            - **data_types**: Columns where data type corrections are necessary (e.g., "datetime", "int").
            - **inconsistent_values**: Columns where values need to be standardized or fixed based on a mapping (e.g., mapping gender values).
            - **outliers**: Columns where outliers should be identified and removed, with a specified threshold (e.g., Z-score threshold).

            Respond ONLY with a JSON format output directly addressing the cleaning recommendations. No json in the start of the string.
            """
        )

        # Anomaly Detection Prompt
        self.anomaly_detection_prompt = PromptTemplate(
            input_variables=['statistical_summary', 'potential_anomalies'],
            template="""
            Anomaly Detection and Suspicious Activity Analysis

            Statistical Summary:
            {statistical_summary}

            Potential Anomalies:
            {potential_anomalies}

            Comprehensive Anomaly Investigation:
            1. Classify and prioritize detected anomalies
            2. Assess potential fraud or data collection errors
            3. Provide contextual explanations
            4. Recommend investigation and mitigation strategies

            Anomaly Assessment Should:
            - Be statistically rigorous
            - Provide clear evidence
            - Offer actionable insights
            - Balance sensitivity and specificity

            Respond ONLY with a JSON format output directly addressing the cleaning recommendations. No json in the start of the string.
            """
        )

        # Transaction Flagging Prompt
        self.transaction_flagging_prompt = PromptTemplate(
            input_variables=['transaction_summary', 'suspicious_indicators'],
            template="""
            Suspicious Transaction Flagging and Analysis

            Transaction Summary:
            {transaction_summary}

            Suspicious Indicators:
            {suspicious_indicators}

            Comprehensive Transaction Risk Assessment:
            1. Identify high-risk transaction patterns
            2. Calculate transaction risk scores
            3. Provide detailed flagging rationale
            4. Recommend immediate actions

            Flagging Process Should:
            - Use multi-dimensional risk assessment
            - Minimize false positives
            - Provide transparent scoring
            - Offer contextual insights

            Respond ONLY with a JSON format output directly addressing the cleaning recommendations. No json in the start of the string.
            """
        )

    def clean_and_validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data cleaning and validation

        :param df: Input pandas DataFrame
        :return: Data cleaning report and cleaned DataFrame
        """
        # Initial data quality checks
        data_issues = {
            'missing_values': df.isnull().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'data_type_issues': df.dtypes.astype(str)
        }

        # Column-level summary
        column_summary = df.describe(include='all').T

        # Create data cleaning chain
        data_cleaning_chain = LLMChain(
            llm=self.llm,
            prompt=self.data_cleaning_prompt
        )

       # Generate data cleaning recommendations
        cleaning_report_str = data_cleaning_chain.run(
            data_issues=str(data_issues),
            column_summary=str(column_summary)
        )

        # Perform basic cleaning
        cleaned_df = df.copy()
        cleaned_df.dropna(inplace=True)
        cleaned_df.drop_duplicates(inplace=True)

        return {
            'cleaning_report': cleaning_report_str,
            'cleaned_dataframe': cleaned_df,
            'data_issues': data_issues
        }

    def detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform statistical anomaly detection

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

        # Anomaly detection chain
        anomaly_detection_chain = LLMChain(
            llm=self.llm,
            prompt=self.anomaly_detection_prompt
        )

        # Generate anomaly detection report
        anomaly_report = anomaly_detection_chain.run(
            statistical_summary=str(statistical_summary),
            potential_anomalies=str(potential_anomalies)
        )

        return {
            'anomaly_report': anomaly_report,
            'potential_anomalies': potential_anomalies,
            'statistical_summary': statistical_summary
        }

    def flag_suspicious_transactions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Flag suspicious transactions based on multiple criteria

        :param df: Input pandas DataFrame
        :return: Transaction flagging report
        """
        sus_df = df.copy(deep=True)
        # Convert 'revenue' column to numeric
        sus_df["revenue"] = sus_df["revenue"].apply(lambda x: int(x.replace("₱", "").replace(",", "")))

        # Define suspicious transaction indicators
        suspicious_indicators = {
            'high_value_transactions': sus_df[sus_df['revenue'] > sus_df['revenue'].quantile(0.95)],
        }

        # Transaction summary
        transaction_summary = sus_df[['revenue', 'transactions', 'device_type']].describe()

        # Transaction flagging chain
        transaction_flagging_chain = LLMChain(
            llm=self.llm,
            prompt=self.transaction_flagging_prompt
        )

        # Generate transaction flagging report
        transaction_report = transaction_flagging_chain.run(
            transaction_summary=str(transaction_summary),
            suspicious_indicators=str(suspicious_indicators)
        )

        return {
            'transaction_report': transaction_report,
            'suspicious_indicators': suspicious_indicators
        }
