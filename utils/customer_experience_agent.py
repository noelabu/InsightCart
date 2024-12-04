import pandas as pd
from typing import Dict
from swarm import  Agent

# Data Processing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class UserSegmentationAgent(Agent):
    def __init__(self):
        def transfer_to_personalization_agent():
            return personalization_agent  # noqa: F821

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
            functions=[transfer_to_personalization_agent]
        )

    def segment_users(self, n_clusters: int = 5, preprocessed_df = pd.DataFrame) -> Dict:
        """
        Perform advanced user segmentation
        
        Args:
            n_clusters (int): Number of user segments
        
        Returns:
            Detailed user segment information
        """
        if preprocessed_df is None:
            raise ValueError("No data has been set. Use set_data() method first.")
        
        df = preprocessed_df.copy()

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
        X_scaled = scaler.fit_transform(self.preprocessed_data[clustering_features])

        # Perform K-Means Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.preprocessed_data['user_segment'] = kmeans.fit_predict(X_scaled)

        # Analyze Segments
        segments = {}
        for segment in range(n_clusters):
            segment_data = self.preprocessed_data[self.preprocessed_data['user_segment'] == segment]

            segments[segment] = {
                'size': len(segment_data),
                'avg_visits': segment_data['visits'].mean(),
                'avg_conversion_rate': segment_data['conversion_rate'].mean(),
                'avg_revenue_per_visit': segment_data['revenue_per_visit'].mean(),
                'source': segment_data['source'].value_counts().to_dict(),
                'device_types': segment_data['device_type'].value_counts().to_dict()
            }

        self.user_segments = segments
        return segments

class PersonalizationAgent(Agent):
    def __init__(self):
        def transfer_to_reporting_agent():
            return reporting_agent  # noqa: F821

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
            functions=[transfer_to_reporting_agent]
        )

    def generate_personalization_strategy(self, segment_data: Dict) -> Dict:
        """
        Generate personalized strategy for a user segment
        
        Args:
            segment_data (Dict): Segment insights
        
        Returns:
            Personalization strategy details
        """
        personalization_strategy = {
            "user_persona": {
                "profile": f"Tech-savvy {segment_data['size']} user segment",
                "key_characteristics": {
                    "avg_visits": segment_data['avg_visits'],
                    "conversion_rate": segment_data['avg_conversion_rate'],
                    "revenue_potential": segment_data['avg_revenue_per_visit']
                }
            },
            "communication_strategy": {
                "primary_channels": list(segment_data['source'].keys()),
                "messaging_tone": "Informative and tech-focused"
            },
            "marketing_recommendations": {
                "top_devices": list(segment_data['device_types'].keys()),
                "targeted_campaigns": "Personalized mobile marketing"
            },
            "conversion_tactics": [
                "Targeted mobile landing pages",
                "Personalized product recommendations",
                "Segment-specific promotional offers"
            ]
        }

        return personalization_strategy

class ReportingAgent(Agent):
    def __init__(self):
        super().__init__(
            name="ReportingAgent",
            instructions="""
            You are a reporting and insights specialist. 
            Compile comprehensive customer experience reports 
            that synthesize segmentation and personalization insights.
            
            Generate clear, actionable, and strategic reports.
            """
        )

    def generate_comprehensive_report(self, segments: Dict, personalization_strategies: Dict) -> str:
        """
        Generate a comprehensive customer experience report
        
        Args:
            segments (Dict): User segment insights
            personalization_strategies (Dict): Personalization strategies
        
        Returns:
            Markdown-formatted report
        """
        report = "# Customer Experience Intelligence Report\n\n"

        for segment_id, segment_data in segments.items():
            report += f"## Segment {segment_id} Analysis\n\n"

            # Segment Statistics
            report += "### Segment Statistics\n"
            for key, value in segment_data.items():
                report += f"- **{key.replace('_', ' ').title()}**: {value}\n"

            # Personalization Strategy
            strategy = personalization_strategies.get(segment_id, {})
            report += "\n### Personalization Strategy\n"
            report += f"```json\n{strategy}\n```\n\n"

        return report
