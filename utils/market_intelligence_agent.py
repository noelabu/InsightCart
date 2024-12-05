from swarm import Agent

class MarketTrendAnalysisAgent(Agent):
    def __init__(self):

        def analyze_market_trends(context_variables):
          """
          Perform market trend analysis

          :param df: Input pandas DataFrame
          :return: Detailed market trend analysis
          """
          # Preprocess data
          traffic_breakdown = context_variables["df"].groupby('source')['transactions'].agg([
              'count',
              'mean',
              ('conversion_rate', lambda x: x.sum() / len(context_variables["df"]) * 100)
          ]).reset_index()

          device_performance = context_variables["df"].groupby('device_type')['revenue'].agg([
              'mean',
              'sum',
              'count'
          ]).reset_index()

          return {
                  "traffic_breakdown": traffic_breakdown.to_dict(orient='records'),
                  "device_performance": device_performance.to_dict(orient='records')
              }

        super().__init__(
            name="MarketTrendAnalysisAgent",
            instructions="""
            You are a market trend analysis specialist.
            Your task is to analyze market data and identify key trends,
            consumer behavior shifts, and emerging opportunities.
            Provide a comprehensive but concise analysis.

            Respond ONLY with a JSON format output directly addressing the analysis in market trends. No json in the start of the string.
            """,
            functions=[analyze_market_trends]
        )

class MarketRecommendationAgent(Agent):
    def __init__(self):

        def recommend_based_on_analysis(context_variables):
          """
          Based on the market trend analysis, provide strategic recommendations.
          """

          return {
                  "market_trend_analysis": context_variables["trend_analysis"],
              }

        super().__init__(
            name="MarketRecommendationAgent",
            model="gpt-4o-mini",
            instructions="""
            You are a strategic recommendations specialist.
            Based on market trend analysis, generate actionable
            strategic recommendations for business growth.

            Provide clear, data-driven insights and prioritized action items.
            """,
            functions=[recommend_based_on_analysis]
        )