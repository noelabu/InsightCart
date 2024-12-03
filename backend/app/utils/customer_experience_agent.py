import os
import pandas as pd
from typing import Dict
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class CustomerExperienceAgent:
    def __init__(self,
                 processed_df: pd.DataFrame,
                 openai_api_key: str,
                 model: str = "gpt-4o-mini"):
        """
        Initialize LangChain-powered Customer Experience Agent

        Args:
            data_path (str): Path to marketplace dataset
            openai_api_key (str): OpenAI API Key
            model (str): LLM model to use
        """
        # Set API Key
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.7,
            max_tokens=1000
        )

        # Load and Preprocess Data
        self.raw_data = processed_df
        self.preprocessed_data = self._preprocess_data()

        # Initialize Components
        self.user_segments = None
        self.insights_vectorstore = None

    def _preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess marketplace data

        Returns:
            Preprocessed DataFrame
        """
        df = self.raw_data.copy()

        # Feature Engineering
        df['conversion_rate'] = df['transactions'] / df['visits']
        df['revenue_per_visit'] = df['revenue'] / df['visits']
        df['engagement_score'] = (df['pageviews'] / df['visits']) * df['conversion_rate']

        return df

    def segment_users(self, n_clusters: int = 5) -> Dict:
        """
        Perform advanced user segmentation

        Args:
            n_clusters (int): Number of user segments

        Returns:
            Detailed user segment information
        """
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

    def create_insights_vectorstore(self):
        """
        Create a vector store of segment insights for semantic retrieval
        """
        # Prepare insights for embedding
        insights_texts = []
        for segment_id, segment_data in self.user_segments.items():
            insight_text = f"""
            Segment {segment_id} Insights:
            - Size: {segment_data['size']}
            - Avg Visits: {segment_data['avg_visits']:.2f}
            - Conversion Rate: {segment_data['avg_conversion_rate']:.2f}
            - Revenue per Visit: {segment_data['avg_revenue_per_visit']:.2f}
            - Top Traffic Sources: {', '.join(segment_data['source'].keys())}
            - Dominant Devices: {', '.join(segment_data['device_types'].keys())}
            """
            insights_texts.append(insight_text)

        # Create Vector Store
        embeddings = OpenAIEmbeddings()
        self.insights_vectorstore = FAISS.from_texts(insights_texts, embeddings)

    def generate_personalization_chain(self) -> LLMChain:
        """
        Create a LangChain for generating personalized strategies

        Returns:
            Personalization LLM Chain
        """
        prompt_template = PromptTemplate(
            input_variables=["segment_data"],
            template="""
            You are a strategic customer experience expert analyzing user segments for an Android phone marketplace.

            Given the following segment data:
            {segment_data}

            Develop a comprehensive personalization strategy that includes:
            1. Detailed User Persona
            2. Communication Strategy
            3. Marketing Channel Recommendations
            4. Conversion Optimization Tactics
            5. Emotional Engagement Approach

            Provide a structured, actionable strategy that addresses the unique characteristics of this user segment.
            """
        )

        return LLMChain(llm=self.llm, prompt=prompt_template)

    def create_customer_experience_agent(self):
        """
        Create a LangChain agent for customer experience analysis

        Returns:
            Initialized LangChain agent
        """
        # Define Tools
        def retrieve_segment_insights(query: str) -> str:
            """Retrieve relevant segment insights"""
            if not self.insights_vectorstore:
                self.create_insights_vectorstore()

            # Semantic search for relevant insights
            results = self.insights_vectorstore.similarity_search(query, k=2)
            return "\n".join([doc.page_content for doc in results])

        tools = [
            Tool(
                name="Segment Insights Retrieval",
                func=retrieve_segment_insights,
                description="Retrieve semantic insights about user segments"
            )
        ]

        # Create Memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Initialize Agent
        agent = initialize_agent(
            tools,
            self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True
        )

        return agent

    def analyze_segment(self, segment_id: int) -> Dict:
        """
        Perform comprehensive analysis of a specific segment

        Args:
            segment_id (int): Target segment to analyze

        Returns:
            Detailed segment analysis
        """
        # Ensure segments are created
        if self.user_segments is None:
            self.segment_users()

        # Get segment data
        segment_data = self.user_segments.get(segment_id, {})

        # Create personalization chain
        personalization_chain = self.generate_personalization_chain()

        # Generate personalized strategy
        strategy = personalization_chain.run(
            segment_data=str(segment_data)
        )

        return {
            "segment_data": segment_data,
            "personalization_strategy": strategy
        }

    def generate_comprehensive_report(self) -> str:
        """
        Generate a comprehensive customer experience report

        Returns:
            Markdown-formatted report
        """
        report = "# Customer Experience Intelligence Report\n\n"

        for segment_id, analysis in self.user_segments.items():
            report += f"## Segment {segment_id} Analysis\n\n"

            # Segment Statistics
            report += "### Segment Statistics\n"
            for key, value in analysis.items():
                report += f"- **{key.replace('_', ' ').title()}**: {value}\n"

            # Personalization Strategy
            segment_analysis = self.analyze_segment(segment_id)
            report += "\n### Personalization Strategy\n"
            report += f"```\n{segment_analysis['personalization_strategy']}\n```\n\n"

        return report
