import streamlit as st
from streamlit_option_menu import option_menu

import json
import pandas as pd
from utils.market_intelligence_agent import MarketTrendAnalysisAgent, MarketRecommendationAgent
from utils.customer_experience_agent import UserSegmentationAgent, PersonalizationAgent
from utils.visual_recommender_agent import DataVisualizerAgent
from utils.plotly_visualization import create_visualizations
from swarm import Swarm

st.set_page_config(page_title="Noela's InsightCart", page_icon="📊", layout="wide")

api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

def generate_visualizations(df, data_report):
    visualizer_agent = DataVisualizerAgent()
    
    # Running the visualization agent
    column_str = ', '.join(df.columns)
    visualization_response = client.run(
        agent=visualizer_agent,
        messages=[
            {"role": "user", "content": f"Recommend visualizations for the report {data_report} and given dataset columns: {column_str}"}
        ],
        context_variables={"report_data":data_report, "dataset_columns": column_str}
    )
    visual_recommendations = visualization_response.messages[-1]["content"]
    visual_recommendations = json.loads(visual_recommendations)
    
    # Generate visualizations
    visualizations = create_visualizations(df, visual_recommendations['recommended_graphs'])

    return visualizations

with st.sidebar:
    page = option_menu(
        "InsightCart",
        ["Home", "About Me", "Market Trend Analysis", "Customer Experience Analysis"],
        icons=['house', 'person-circle', 'graph-up', 'people'],
        menu_icon="list",
        default_index=0,
    )

if not api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar to use the application.")

else:
    if page == "Home":
        st.title("InsightCart")
        st.write("""
        InsightCart is an intelligent data analytics platform designed to empower businesses with comprehensive insights into their datasets. 
        It is built around a suite of specialized agents, each tailored to handle a specific aspect of data analysis, processing, marketing intelligence, and customer experience. 
        The platform helps organizations clean and process their data, detect anomalies, assess transaction risks, analyze market trends, personalize user experiences, and generate insightful reports and visual recommendations.
        """)
        st.subheader("The app is composed of the following key Agents:")
        st.markdown("### 1. Data Detective")
        st.write("**Purpose:** The Data Detective agent focuses on the initial steps of data cleaning and anomaly detection.")
        st.write("""
        - **Data Cleaning:** Identifies and removes inaccuracies, duplicates, or irrelevant data points from the dataset.
        - **Anomaly Detection:** Detects unusual patterns or outliers in the data that may indicate errors, fraud, or other issues.
        - **Transaction Risk Assessment:** Evaluates the dataset for potential risks associated with transactions, such as suspicious activity or high-risk patterns.
        """)
        st.markdown("### 2. Data Processing")
        st.write("**Purpose:** The Data Processing agent leverages the cleaning and anomaly reports to prepare the dataset for deeper analysis.")
        st.write("""
        - **Data Transformation:** Transforms the cleaned and validated dataset into a structure that is suitable for advanced analytics and reporting.
        - **Data Enrichment:** Combines external data sources with internal datasets to provide a richer, more comprehensive view of the business environment.
        """)
        st.markdown("### 3. Marketing Intelligence")
        st.write("**Purpose:** The Marketing Intelligence suite consists of two agents that help businesses understand market trends and generate tailored recommendations.")
        st.write("""
        - **Market Trend Analysis Agent:** Analyzes current and historical market data to identify trends, emerging patterns, and shifts in the market landscape.
        - **Market Recommendation Agent:** Based on trend analysis, this agent provides recommendations for businesses to adapt to market conditions, including product adjustments, pricing strategies, and potential new market segments.
        """)
        st.markdown("### 4. Customer Experience")
        st.write("**Purpose:** This suite of agents focuses on segmenting users and personalizing their experiences.")
        st.write("""
        - **User Segmentation Agent:** Segments users within the dataset based on their behavior, demographics, or other relevant attributes, enabling more targeted analysis.
        - **Personalization Agent:** Uses the segmented data to generate personalized strategies aimed at enhancing user experience, improving engagement, and optimizing product offerings.
        - **Reporting Agent:** Generates detailed reports on user segments and personalization strategies, offering insights into how the strategies are performing and where improvements can be made.
        """)
        st.markdown("### 5. Visual Recommender")
        st.write("**Purpose:** This agent provides businesses with visual recommendations based on the generated reports.")
        st.write("- **Graph Recommendations:** Analyzes the data in the reports and suggests appropriate visualizations (e.g., bar charts, heat maps, line graphs) to represent the key insights and make them more accessible to stakeholders.")
        st.subheader("Key Features")
        st.write("""
        - **Comprehensive Data Cleaning:** Automatically detect and address issues in raw datasets to ensure quality data.
        - **Advanced Anomaly Detection:** Identify patterns and anomalies that could indicate fraud, errors, or unexpected behaviors.
        - **Transaction Risk Management:** Assess the risks associated with transactions to safeguard against financial losses.
        - **Market Trend Analysis:** Stay on top of current market trends and anticipate shifts in demand, competition, and consumer behavior.
        - **Personalized User Experience:** Automatically create targeted user segments and deliver customized strategies for improved engagement.
        - **Actionable Reporting:** Generate clear, concise reports that detail user segments, personalization strategies, and market recommendations.
        - **Intelligent Visualization:** Get recommendations for the best types of visualizations to communicate your insights effectively.
        """)
        st.subheader("How It Works")
        st.write("""
        1. **Data Intake:** Users upload their raw datasets into the platform.
        2. **Data Detective:** The Data Detective agent scans the dataset for errors, anomalies, and transaction risks.
        3. **Data Processing:** After cleaning and anomaly detection, the Data Processing agent transforms the dataset into a usable format for deeper analysis.
        4. **Marketing Intelligence:** The Market Trend Analysis and Market Recommendation agents analyze market data and suggest adaptive strategies.
        5. **Customer Experience:** The User Segmentation agent segments users, the Personalization agent creates strategies for each segment, and the Reporting agent generates comprehensive reports.
        6. **Visual Recommender:** The Visual Recommender agent suggests relevant graphs and charts based on the reports to help users visualize their insights.
        """)
                
    elif page == "About Me":
        st.header("About Me")
        st.markdown("""
        Hi! I'm Noela Jean Bunag, a Python Developer and AI Enthusiast. I'm passionate about creating accessible AI solutions and exploring the possibilities of Natural Language Processing.
        
        Connect with me on [LinkedIn](https://www.linkedin.com/in/noela-bunag/) to discuss AI, Python development, or potential collaborations.
        
        Check out my portfolio at [noelabu.github.io](https://noelabu.github.io/) to see more of my projects and work.
        """)

    elif page == "Market Trend Analysis":
        st.title("Market Trend Analysis")

        # Client initialization
        client = Swarm()

        # Load the data outside of the callback
        df = pd.read_csv('https://raw.githubusercontent.com/noelabu/InsightCart/refs/heads/develop/data/sales_data_cleaned.csv')

        # Create agents
        market_recommendation_agent = MarketRecommendationAgent()
        trend_analysis_agent = MarketTrendAnalysisAgent()
        

        # Running the trend analysis agent
        response = client.run(
            agent=trend_analysis_agent,
            messages=[{"role": "user", "content": "Analyze market trends for our digital marketing data."}],
            context_variables={"df": df}
        )
        trend_analysis = response.messages[-1]["content"]
        
        # Running the market recommendation agent
        recommendation_response = client.run(
            agent=market_recommendation_agent,
            messages=[{"role": "user", "content": "Generate strategic recommendations based on the trend analysis"}],
            context_variables={"trend_analysis": trend_analysis}
        )
        strategic_recommendations = recommendation_response.messages[-1]["content"]
        
        market_trend_visualizations = generate_visualizations(df, trend_analysis)
        for title, visualization_data in market_trend_visualizations.items():
            st.plotly_chart(visualization_data['figure'])

        st.markdown(strategic_recommendations)
    
    elif page == "Customer Experience Analysis":
        st.title("Customer Experience")

        # Client initialization
        client = Swarm()

        # Load the data outside of the callback
        df = pd.read_csv('https://raw.githubusercontent.com/noelabu/InsightCart/refs/heads/develop/data/sales_data_cleaned.csv')

        # Create agents
        personalization_agent = PersonalizationAgent()
        segmentation_agent = UserSegmentationAgent()
        visualizer_agent = DataVisualizerAgent()

        # Run Segmentation Workflow
        segmentation_response = client.run(
            agent=segmentation_agent,
            messages=[
                {"role": "user", "content": "Perform advanced user segmentation"},
            ],
            context_variables={"df": df}
        )

        # Extract segments from response
        segments = segmentation_response.messages[-1]["content"]
        
        # Run Personalization Workflow
        personalization_response = client.run(
            agent=personalization_agent,
            messages=[
                {"role": "user", "content": "Generate personalization strategies for user segments"}
            ],
            context_variables={"segment_data": segments}
        )

        # Extract personalization strategies
        personalization_strategies = personalization_response.messages[-1]["content"]
        
        customer_experience_visualizations = generate_visualizations(df, personalization_strategies)
        for title, visualization_data in customer_experience_visualizations.items():
            st.plotly_chart(visualization_data['figure'])
        
        st.markdown(personalization_strategies)
        
        