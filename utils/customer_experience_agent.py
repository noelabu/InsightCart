from typing import Dict
from swarm import  Agent

# Data Processing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class UserSegmentationAgent(Agent):
    def __init__(self):
        def segment_users(context_variables) -> Dict:
          """
          Perform advanced user segmentation

          Returns:
              Detailed user segment information
          """

          n_clusters = 5
          df = context_variables["df"]

          # Feature Engineering
          df['conversion_rate'] = df['transactions'] / df['visits']
          df['revenue_per_visit'] = df['revenue'] / df['visits']
          df['engagement_score'] = (df['pageviews'] / df['visits']) * df['conversion_rate']

          # Select clustering features
          clustering_features = [
              'visits', 'pageviews', 'transactions',
              'conversion_rate', 'revenue_per_visit'
          ]

          # Normalize features
          scaler = StandardScaler()
          X_scaled = scaler.fit_transform(df[clustering_features])

          # Perform K-Means Clustering
          kmeans = KMeans(n_clusters=n_clusters, random_state=42)
          df['user_segment'] = kmeans.fit_predict(X_scaled)

          # Analyze Segments
          segments = {}
          for segment in range(n_clusters):
              segment_data = df[df['user_segment'] == segment]

              segments[segment] = {
                  'size': len(segment_data),
                  'avg_visits': segment_data['visits'].mean(),
                  'avg_conversion_rate': segment_data['conversion_rate'].mean(),
                  'avg_revenue_per_visit': segment_data['revenue_per_visit'].mean(),
                  'source': segment_data['source'].value_counts().to_dict(),
                  'device_types': segment_data['device_type'].value_counts().to_dict()
              }
          return segments


        super().__init__(
            name="UserSegmentationAgent",
            instructions="""
            You are a user segmentation specialist responsible for:
            1. Preprocessing marketplace data
            2. Performing advanced user clustering
            3. Generating detailed user segment insights
            4. Preparing data for personalization strategy

            Respond ONLY with a JSON format output directly addressing the segmented data. No json in the start of the string.
            """,
            functions=[segment_users]
        )


class PersonalizationAgent(Agent):
    def __init__(self):
      def generate_personalization_strategy(context_variables) -> Dict:
        """
        Generate personalized strategy for a user segment
        """
        return {
                "user_segment_data": context_variables["segment_data"],
            }

      super().__init__(
            name="PersonalizationAgent",
            instructions="""
            You are a customer experience personalization strategist.
            Your role is to:
            1. Develop personalized strategies for each user segment
            2. Create targeted communication and marketing approaches
            3. Generate actionable insights for improving customer experience

            Respond ONLY with a JSON format output directly addressing the personalization strategy. No json in the start of the string.
            """,
            functions=[generate_personalization_strategy]
        )

class ReportingAgent(Agent):
  def __init__(self):
    def generate_comprehensive_report(context_variables) -> str:
      """
      Generate a comprehensive customer experience report
      """
      return {
          "segments": context_variables["segments"],
          "personalization_strategies": context_variables["personalization_strategies"]
      }

    super().__init__(
      name="ReportingAgent",
      instructions="""
      You are a reporting and insights specialist.
      Compile comprehensive customer experience reports
      that summarize user segment statistics and corresponding personalization strategies.

      Generate clear, actionable, and strategic reports based on the segments and personalization strategies of each segment.
      """,
      function=[generate_comprehensive_report]
    )
