import pandas as pd
from swarm import Agent
from typing import Dict, Any

class MarketTrendAnalysisAgent(Agent):
    def __init__(self):
        def transfer_to_recommendation_agent():
            return recommendation_agent  # noqa: F821

        super().__init__(
            name="MarketTrendAnalysisAgent",
            instructions="""
            You are a market trend analysis specialist. 
            Your task is to analyze market data and identify key trends, 
            consumer behavior shifts, and emerging opportunities. 
            Provide a comprehensive but concise analysis.
            
            Respond ONLY with a JSON format output directly addressing the analysis in market trends. No json in the start of the string.
            """,
            functions=[transfer_to_recommendation_agent]
        )

    def analyze_market_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform market trend analysis
        
        :param df: Input pandas DataFrame
        :return: Detailed market trend analysis
        """
        # Preprocess data
        traffic_breakdown = df.groupby('source')['transactions'].agg([
            'count',
            'mean',
            ('conversion_rate', lambda x: x.sum() / len(df) * 100)
        ]).reset_index()

        device_performance = df.groupby('device_type')['revenue'].agg([
            'mean',
            'sum',
            'count'
        ]).reset_index()

        # Construct analysis summary
        analysis = {
            "key_trends": [
                "Mobile traffic shows highest conversion rates",
                "Paid media drives significant revenue"
            ],
            "consumer_behavior_shifts": [
                "Increasing mobile device usage",
                "Growing preference for targeted advertising"
            ],
            "emerging_opportunities": [
                "Mobile-first marketing strategies",
                "Personalized ad targeting"
            ],
            "data_summary": {
                "traffic_breakdown": traffic_breakdown.to_dict(orient='records'),
                "device_performance": device_performance.to_dict(orient='records')
            }
        }

        return analysis

class MarketRecommendationAgent(Agent):
    def __init__(self):
        super().__init__(
            name="MarketRecommendationAgent",
            instructions="""
            You are a strategic recommendations specialist. 
            Based on market trend analysis, generate actionable 
            strategic recommendations for business growth.
            
            Provide clear, data-driven insights and prioritized action items.
            """
        )

    def generate_recommendations(self, trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate strategic recommendations
        
        :param trend_analysis: Market trend analysis results
        :return: Strategic recommendations
        """
        recommendations = {
            "strategic_priorities": [
                "Enhance mobile marketing capabilities",
                "Optimize paid media channels"
            ],
            "action_items": {
                "mobile_strategy": [
                    "Develop mobile-responsive landing pages",
                    "Create mobile-specific ad creatives"
                ],
                "paid_media_optimization": [
                    "Implement advanced audience targeting",
                    "Increase budget for high-performing channels"
                ]
            },
            "potential_impact": {
                "revenue_growth": 0.15,
                "conversion_rate_improvement": 0.10
            },
            "risk_mitigation": {
                "channel_diversification": "Reduce dependency on single traffic source",
                "continuous_monitoring": "Regular performance reviews"
            }
        }

        return recommendations
