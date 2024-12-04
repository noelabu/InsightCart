import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

class PlotlyVisualization:
    def __init__(self, dataset, recommended_graphs):
        """
        Initialize the visualization agent with dataset and graph recommendations.

        :param dataset: pandas DataFrame containing the data
        :param recommended_graphs: List of graph specifications
        """
        self.dataset = dataset
        self.recommended_graphs = recommended_graphs

    def validate_columns(self, graph_spec):
        """
        Validate that required columns exist in the dataset.

        :param graph_spec: Dictionary containing graph specification
        :return: Boolean indicating column availability
        """
        columns_needed = graph_spec.get('columns_needed', [])
        return all(col in self.dataset.columns for col in columns_needed)

    def generate_visualizations(self):
        """
        Generate all recommended visualizations.

        :return: Dictionary of graph specifications and their corresponding Plotly figures
        """
        visualizations = {}

        for graph_spec in self.recommended_graphs:
            # Validate columns before attempting to create visualization
            if not self.validate_columns(graph_spec):
                print(f"Skipping {graph_spec['title']} - Required columns not found")
                continue

            # Create visualization based on graph type
            try:
                if graph_spec['type'] == 'bar chart':
                    fig = self._create_bar_chart(graph_spec)
                elif graph_spec['type'] == 'scatter plot':
                    fig = self._create_scatter_plot(graph_spec)
                elif graph_spec['type'] == 'line chart':
                    fig = self._create_line_chart(graph_spec)
                elif graph_spec['type'] == 'box plot':
                    fig = self._create_boxplot(graph_spec)
                elif graph_spec['type'] == 'histogram':
                    fig = self._create_histogram(graph_spec)
                else:
                    print(f"Unsupported graph type: {graph_spec['type']}")
                    continue

                visualizations[graph_spec['title']] = {
                    'figure': fig,
                    'description': graph_spec.get('description', '')
                }
            except Exception as e:
                print(f"Error creating {graph_spec['title']}: {str(e)}")


        return visualizations

    def _create_bar_chart(self, graph_spec):
        """
        Create a bar chart based on graph specification.

        :param graph_spec: Dictionary containing bar chart specification
        :return: Plotly Figure object
        """
        # Support for color on x-axis
        color = graph_spec['x_axis']

        if color:
            fig = px.bar(
                self.dataset,
                x=graph_spec['x_axis'],
                y=graph_spec['y_axis'],
                color=color,
                title=graph_spec['title']
            )
        else:
            fig = px.bar(
                self.dataset,
                x=graph_spec['x_axis'],
                y=graph_spec['y_axis'],
                title=graph_spec['title']
            )

        fig.update_layout(
            xaxis_title=graph_spec.get('x_axis_title', graph_spec['x_axis']),
            yaxis_title=graph_spec.get('y_axis_title', graph_spec['y_axis'])
        )
        return fig

    def _create_scatter_plot(self, graph_spec):
        """
        Create a scatter plot based on graph specification.

        :param graph_spec: Dictionary containing scatter plot specification
        :return: Plotly Figure object
        """
        # Support for color on x-axis and additional size parameter
        color = graph_spec['x_axis']

        if color:
            fig = px.scatter(
                self.dataset,
                x=graph_spec['x_axis'],
                y=graph_spec['y_axis'],
                color=color,
                title=graph_spec['title']
            )
        else:
            fig = px.scatter(
                self.dataset,
                x=graph_spec['x_axis'],
                y=graph_spec['y_axis'],
                title=graph_spec['title']
            )

        fig.update_layout(
            xaxis_title=graph_spec.get('x_axis_title', graph_spec['x_axis']),
            yaxis_title=graph_spec.get('y_axis_title', graph_spec['y_axis'])
        )
        return fig

    def _create_line_chart(self, graph_spec):
        """
        Create a line chart based on graph specification.

        :param graph_spec: Dictionary containing line chart specification
        :return: Plotly Figure object
        """
        # Support for color on x-axis
        color = graph_spec['x_axis']

        if color:
            fig = px.line(
                self.dataset,
                x=graph_spec['x_axis'],
                y=graph_spec['y_axis'],
                color=color,
                title=graph_spec['title']
            )
        else:
            fig = px.line(
                self.dataset,
                x=graph_spec['x_axis'],
                y=graph_spec['y_axis'],
                title=graph_spec['title']
            )

        fig.update_layout(
            xaxis_title=graph_spec.get('x_axis_title', graph_spec['x_axis']),
            yaxis_title=graph_spec.get('y_axis_title', graph_spec['y_axis'])
        )
        return fig
    
    def _create_boxplot(self, graph_spec):
        """
        Create a boxplot based on graph specification.

        :param graph_spec: Dictionary containing boxplot specification
        :return: Plotly Figure object
        """
        # Support for color grouping
        color = graph_spec.get('color')

        if color:
            fig = px.box(
                self.dataset,
                x=graph_spec.get('x_axis'),
                y=graph_spec['y_axis'],
                color=color,
                title=graph_spec['title']
            )
        else:
            fig = px.box(
                self.dataset,
                x=graph_spec.get('x_axis'),
                y=graph_spec['y_axis'],
                title=graph_spec['title']
            )

        fig.update_layout(
            xaxis_title=graph_spec.get('x_axis_title', graph_spec.get('x_axis', '')),
            yaxis_title=graph_spec.get('y_axis_title', graph_spec['y_axis'])
        )
        return fig

    def _create_histogram(self, graph_spec):
        """
        Create a histogram based on graph specification.

        :param graph_spec: Dictionary containing histogram specification
        :return: Plotly Figure object
        """
        # Support for color grouping and additional parameters
        color = graph_spec.get('color')
        
        if color:
            fig = px.histogram(
                self.dataset,
                x=graph_spec['x_axis'],
                color=color,
                title=graph_spec['title'],
                marginal=graph_spec.get('marginal'),  # Optional: 'rug', 'box', 'violin'
                nbins=graph_spec.get('nbins')  # Optional: number of bins
            )
        else:
            fig = px.histogram(
                self.dataset,
                x=graph_spec['x_axis'],
                title=graph_spec['title'],
                marginal=graph_spec.get('marginal'),
                nbins=graph_spec.get('nbins')
            )

        fig.update_layout(
            xaxis_title=graph_spec.get('x_axis_title', graph_spec['x_axis']),
            yaxis_title=graph_spec.get('y_axis_title', 'Count')
        )
        return fig


def create_visualizations(dataset, recommended_graphs):
    """
    Convenience function to create visualizations from a dataset and graph specifications.

    :param dataset: pandas DataFrame
    :param recommended_graphs: List of graph specifications
    :return: Dictionary of generated visualizations
    """
    agent = PlotlyVisualization(dataset, recommended_graphs)
    visualizations = agent.generate_visualizations()
    return visualizations