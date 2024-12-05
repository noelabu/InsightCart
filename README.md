# InsightCart

InsightCart is an intelligent data analytics platform designed to empower businesses with comprehensive insights into their datasets. It is built around a suite of specialized agents, each tailored to handle a specific aspect of data analysis, processing, marketing intelligence, and customer experience. The platform helps organizations clean and process their data, detect anomalies, assess transaction risks, analyze market trends, personalize user experiences, and generate insightful reports and visual recommendations.

The app is composed of the following key agents:

### 1. **Data Detective**
   - **Purpose**: The Data Detective agent focuses on the initial steps of data cleaning and anomaly detection.
     - **Data Cleaning**: Identifies and removes inaccuracies, duplicates, or irrelevant data points from the dataset.
     - **Anomaly Detection**: Detects unusual patterns or outliers in the data that may indicate errors, fraud, or other issues.
     - **Transaction Risk Assessment**: Evaluates the dataset for potential risks associated with transactions, such as suspicious activity or high-risk patterns.

### 2. **Data Processing**
   - **Purpose**: The Data Processing agent leverages the cleaning and anomaly reports to prepare the dataset for deeper analysis.
     - **Data Transformation**: Transforms the cleaned and validated dataset into a structure that is suitable for advanced analytics and reporting.
     - **Data Enrichment**: Combines external data sources with internal datasets to provide a richer, more comprehensive view of the business environment.

### 3. **Marketing Intelligence**
   - **Purpose**: The Marketing Intelligence suite consists of two agents that help businesses understand market trends and generate tailored recommendations.
     - **Market Trend Analysis Agent**: Analyzes current and historical market data to identify trends, emerging patterns, and shifts in the market landscape.
     - **Market Recommendation Agent**: Based on trend analysis, this agent provides recommendations for businesses to adapt to market conditions, including product adjustments, pricing strategies, and potential new market segments.

### 4. **Customer Experience**
   - **Purpose**: This suite of agents focuses on segmenting users and personalizing their experiences.
     - **User Segmentation Agent**: Segments users within the dataset based on their behavior, demographics, or other relevant attributes, enabling more targeted analysis.
     - **Personalization Agent**: Uses the segmented data to generate personalized strategies aimed at enhancing user experience, improving engagement, and optimizing product offerings.
     - **Reporting Agent**: Generates detailed reports on user segments and personalization strategies, offering insights into how the strategies are performing and where improvements can be made.

### 5. **Visual Recommender**
   - **Purpose**: This agent provides businesses with visual recommendations based on the generated reports.
     - **Graph Recommendations**: Analyzes the data in the reports and suggests appropriate visualizations (e.g., bar charts, heat maps, line graphs) to represent the key insights and make them more accessible to stakeholders.

---

## Key Features

- **Comprehensive Data Cleaning**: Automatically detect and address issues in raw datasets to ensure quality data.
- **Advanced Anomaly Detection**: Identify patterns and anomalies that could indicate fraud, errors, or unexpected behaviors.
- **Transaction Risk Management**: Assess the risks associated with transactions to safeguard against financial losses.
- **Market Trend Analysis**: Stay on top of current market trends and anticipate shifts in demand, competition, and consumer behavior.
- **Personalized User Experience**: Automatically create targeted user segments and deliver customized strategies for improved engagement.
- **Actionable Reporting**: Generate clear, concise reports that detail user segments, personalization strategies, and market recommendations.
- **Intelligent Visualization**: Get recommendations for the best types of visualizations to communicate your insights effectively.

---

## How It Works

1. **Data Intake**: Users upload their raw datasets into the platform.
2. **Data Detective**: The Data Detective agent scans the dataset for errors, anomalies, and transaction risks.
3. **Data Processing**: After cleaning and anomaly detection, the Data Processing agent transforms the dataset into a usable format for deeper analysis.
4. **Marketing Intelligence**: The Market Trend Analysis and Market Recommendation agents analyze market data and suggest adaptive strategies.
5. **Customer Experience**: The User Segmentation agent segments users, the Personalization agent creates strategies for each segment, and the Reporting agent generates comprehensive reports.
6. **Visual Recommender**: The Visual Recommender agent suggests relevant graphs and charts based on the reports to help users visualize their insights.

---

## Getting Started

To run **InsightCart** locally, clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/noelabu/InsightCart.git
cd InsightCart
pip install -r requirements.txt
```

### API Key Configuration

#### 1. **OpenAI API Key**  
To use the OpenAI API for advanced route recommendations, you'll need an API key. Sign up at [OpenAI](https://openai.com) and create an API key. Store your key in an environment variable named `OPENAI_API_KEY`.

```bash
export OPENAI_API_KEY='your-openai-api-key'
```

### Running the Streamlit App
Once you've set up the API key, you can run the Streamlit app to interact with **InisghtCart**.

```bash
streamlit run app.py
```

Open your browser and go to `http://localhost:8501` to access the application. 

---
## Conclusion

InsightCart offers a powerful suite of AI-powered agents designed to help businesses extract actionable insights from their data. Whether you're focused on data cleaning, market analysis, user segmentation, or personalized recommendations, InsightCart equips you with the tools necessary to make informed decisions and drive business growth.
