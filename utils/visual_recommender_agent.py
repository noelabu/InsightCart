from typing import Dict
from swarm import Agent

class DataVisualizerAgent(Agent):
    def __init__(self):
        def recommend_graphs(context_variables) -> Dict:
          """
          Recommend the most suitable visualizations
          """
          return {
              "report_data": context_variables["report_data"],
              "dataset_columns": context_variables["dataset_columns"]
          }

        super().__init__(
            name="DataVisualizerAgent",
            instructions="""
              You are a data visualization expert. Analyze the following report data
              and recommend the most suitable visualization graphs. For each recommendation,
              provide:
              - Graph type (bar chart, scatter plot, etc.)
              - X-axis column
              - Y-axis column
              - Columns needed (only consider the columns that is in the dataset columns)
              - Brief description of why this visualization is appropriate

              Respond strictly in the following JSON format. No json in the start of the string.
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
              """,
            functions=[recommend_graphs]
        )
