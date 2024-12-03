import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Dict, Any

class MarketIntelligenceAgent:
    def __init__(self, openai_api_key):
        """
        Initialize the Market Intelligence LLM Agent

        :param openai_api_key: OpenAI API key for authentication
        """
        # Set up the primary language model
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=2000,
            api_key=openai_api_key
        )

        # Prepare comprehensive prompt templates
        self.trend_analysis_prompt = PromptTemplate(
            input_variables=['data_summary', 'time_period'],
            template="""
            Market Trend Analysis Report for Android Phone Marketplace

            Based on the following data summary:
            {data_summary}

            Perform a comprehensive market trend analysis for the {time_period} period:
            1. Identify key market trends
            2. Analyze significant shifts in consumer behavior
            3. Highlight potential emerging opportunities
            4. Provide strategic recommendations for business growth

            Your analysis should be:
            - Data-driven
            - Forward-looking
            - Actionable
            - Concise yet comprehensive

            Respond ONLY with a JSON format output directly addressing the cleaning recommendations. No json in the start of the string.
            """
        )

        # Create LLM Chain for trend analysis
        self.trend_analysis_chain = LLMChain(
            llm=self.llm,
            prompt=self.trend_analysis_prompt
        )

    def _preprocess_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Preprocess and summarize dataset for LLM analysis

        :param df: Input pandas DataFrame
        :return: Dictionary of summarized insights
        """
        # Traffic source analysis
        traffic_breakdown = df.groupby('source')['transactions'].agg([
            'count',
            'mean',
            ('conversion_rate', lambda x: x.sum() / len(df) * 100)
        ]).reset_index()

        # Device type performance
        device_performance = df.groupby('device_type')['revenue'].agg([
            'mean',
            'sum',
            'count'
        ]).reset_index()

        # Campaign type effectiveness
        campaign_effectiveness = df.groupby('medium')['revenue'].agg([
            'mean',
            'sum',
            ('roas', lambda x: x.sum() / df[df['medium'] == x.name]['ad spend'].sum())
        ]).reset_index()

        # Aggregate data summary
        data_summary = f"""
        Traffic Source Breakdown:
        {traffic_breakdown.to_string()}

        Device Performance:
        {device_performance.to_string()}

        Campaign Effectiveness:
        {campaign_effectiveness.to_string()}
        """

        return {
            'data_summary': data_summary,
            'traffic_breakdown': traffic_breakdown,
            'device_performance': device_performance,
            'campaign_effectiveness': campaign_effectiveness
        }

    def analyze_market_trends(self, df: pd.DataFrame, time_period: str = 'Quarterly') -> str:
        """
        Perform comprehensive market trend analysis

        :param df: Input pandas DataFrame
        :param time_period: Analysis time period
        :return: Detailed market trend report
        """
        # Preprocess data
        processed_data = self._preprocess_data(df)

        # Generate trend analysis report
        trend_report = self.trend_analysis_chain.run(
            data_summary=processed_data['data_summary'],
            time_period=time_period
        )

        return trend_report

    def generate_strategic_recommendations(self, trend_report: Dict) -> str:
        """
        Generate strategic recommendations based on trend analysis

        :param trend_report: Market trend report
        :return: Strategic recommendations
        """
        strategic_prompt = PromptTemplate(
            input_variables=['trend_report'],
            template="""
            Analyze the following market trend report:
            {trend_report}

            Develop a comprehensive strategic roadmap:
            1. Identify top 3 strategic priorities
            2. Propose specific action items for each priority
            3. Estimate potential business impact
            4. Outline potential risks and mitigation strategies

            Recommendations should be:
            - Actionable
            - Data-driven
            - Aligned with market trends

            """)

        strategic_chain = LLMChain(llm=self.llm, prompt=strategic_prompt)
        strategic_recommendations = strategic_chain.run(trend_report=trend_report)

        return strategic_recommendations