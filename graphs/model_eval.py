import pandas as pd
import plotly.graph_objects as go

# The data dictionary as provided
data = {
    "Models Trained": ["r12", "r24", "r25"],
    "Market1501": {
        "mAP": [39.48, 44.6, 38.9],
        "Rank-1": [66.1, 68.5, 63.4],
        "Rank-5": [83.9, 84.6, 81.7]
    },
    "DukeMTMC": {
        "mAP": [10.9, 24.1, 23.3],
        "Rank-1": [22.2, 43.9, 45.01],
        "Rank-5": [36.7, 61.1, 63.8]
    }
}

# Function to create a Plotly line graph based on user input
def create_plotly_line_graph(rows, metrics, datasets):
    # Filter rows (subtract 1 for zero-indexed DataFrame)
    rows = [r-1 for r in rows if r-1 in range(len(data["Models Trained"]))]
    
    # Initialize a figure
    fig = go.Figure()

    # Create traces for each selected metric
    for metric in metrics:
        for dataset in datasets:
            # Prepare the metric data for the selected rows
            metric_data = [data[dataset][metric][r] for r in rows]
            # Add a trace for this metric
            fig.add_trace(go.Scatter(
                x=[data["Models Trained"][r] for r in rows],
                y=metric_data,
                mode='lines+markers',
                name=f'{dataset} {metric}'
            ))

    # Update the layout of the figure
    fig.update_layout(
        title='Evaluation Metrics Line Graph',
        xaxis_title='Models Trained',
        yaxis_title='Metric Values',
        legend_title='Metrics'
    )

    return fig

# User inputs for rows, metrics, and datasets to plot
rows = [1, 2, 3]  # Rows from the spreadsheet
metrics = ["Rank-5"]  # Metrics to plot
datasets = ["Market1501", "DukeMTMC"]  # Datasets to plot

# Create the line graph
fig_line = create_plotly_line_graph(rows, metrics, datasets)

# Show the figure
fig_line.show()
