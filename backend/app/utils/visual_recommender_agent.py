import json
from typing import Dict, Any
import openai

class DataVisualizerAgent:
    def __init__(self, api_key: str):
        """
        Initialize the LLM-based Data Visualizer

        :param api_key: OpenAI API key for GPT model access
        """
        openai.api_key = api_key

    def analyze_report(self, report_data: Dict[str, Any], column_str) -> Dict[str, Any]:
        """
        Use LLM to analyze report and recommend visualizations

        :param report_data: Input report data to be visualized
        :return: Recommended visualization graphs
        """
        # Convert report data to a string for LLM input
        report_str = json.dumps(report_data, indent=2)

        # Prompt for visualization recommendation
        prompt = f"""
        You are a data visualization expert. Analyze the following report data
        and recommend the most suitable visualization graphs. For each recommendation,
        provide:
        - Graph type (bar chart, scatter plot, etc.)
        - X-axis column
        - Y-axis column
        - Columns needed (only the columns in the Dataset Columns)
        - Brief description of why this visualization is appropriate

        Report Data:
        {report_str}

        Dataset Columns:
        {column_str}

        Respond strictly in the following JSON format:
        {{
            "recommended_graphs": [
                {{
                    "type": "string",
                    "x_axis": "string",
                    "y_axis": "string",
                    "columns_needed": ["string"],
                    "title": "string",
                    "description": "string"
                }}
            ]
        }}
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data visualization expert."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )

            # Parse and return the LLM's recommendations
            recommendations = json.loads(response.choices[0].message.content)
            return recommendations

        except Exception as e:
            return {
                "error": str(e),
                "recommendations": []
            }