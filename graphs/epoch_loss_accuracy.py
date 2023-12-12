import json, os, gspread, requests, datetime, time
from oauth2client.service_account import ServiceAccountCredentials

import plotly.graph_objects as go
from plotly.subplots import make_subplots

sample_data = {
        "epoch_1": {"loss": "4.3745", "acc": "22.9796"},
        "epoch_2": {"loss": "2.2409", "acc": "66.1936"},
        "epoch_3": {"loss": "1.6343", "acc": "83.3572"},    
}

def _get_worksheet():
    try:
        EXCEL_LINK = "https://docs.google.com/spreadsheets/d/1qtLI_GLpcnPONtLXDg56aBfNlp5r1jlSMQ5QORbuBVs/edit?usp=sharing"
        KEY_FILE = "./excel-service-key.json"

        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        credentials = ServiceAccountCredentials.from_json_keyfile_name(KEY_FILE, scope)
        gc = gspread.authorize(credentials)

        document = gc.open_by_url(EXCEL_LINK)
        worksheet = document.worksheet(WORKSHEET_NAME)

        return worksheet
    except Exception as e:
        print('exception: ', e)

def _save_graph(fig): 
    # Save the figure as an HTML file and open in browser
    directory = f"./training_results/{WORKSHEET_NAME}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f'r{TARGET_ROW}-epoch-loss.html')
    fig.write_html(file_path, auto_open=True)

def plot_graph(data):
        # Extract loss and accuracy values
    epochs = [int(epoch.split("_")[1]) for epoch in data.keys()]
    loss_values = [float(data[epoch]["loss"]) for epoch in data.keys()]
    acc_values = [float(data[epoch]["acc"]) for epoch in data.keys()]

    # Define custom colors for the plots
    loss_color = "red"
    acc_color = "green"

    # Create separate figures
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        subplot_titles=("Loss Over Epochs", "Accuracy Over Epochs"),
    )

    # Add Loss plot
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=loss_values,
            mode="lines+markers",
            name="Loss",
            line=dict(color=loss_color),
        ),
        row=1,
        col=1,
    )

    # Add Accuracy plot
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=acc_values,
            mode="lines+markers",
            name="Accuracy",
            line=dict(color=acc_color),
        ),
        row=2,
        col=1,
    )
    fig.update_traces(hoverlabel=dict(font_color='white'))
    # Update layout for each figure
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=2, col=1)
    fig.update_layout(title="Training Loss, Accuracy over epochs",
    plot_bgcolor='#f7f7f7',
    annotations=[
        dict(
            text=f"{WORKSHEET_NAME} - R{TARGET_ROW}",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.006,  
            y=1.05, 
            xanchor="left",
            yanchor="top",  
            font=dict(
                size=14,
                color="black"
            )
        )
    ],
    legend=dict(
            orientation="h",  # "h" for horizontal, "v" for vertical
            y=-0.08,  # Adjust this value to move the legend below the graph
            font=dict(
                size=16,  # Adjust this value to increase the font size
                color="black",  # You can also change the font color if needed
                # family="Arial"  # Optional: You can specify the font family
            ),
        )
    )

    _save_graph(fig)

WORKSHEET_NAME = "Finetune with RP - ResNet18"
TARGET_ROW = 7

worksheet = _get_worksheet()
epoch_logs = worksheet.acell(f'E{TARGET_ROW}')

if isinstance(epoch_logs.value, str):
    epoch_logs = json.loads(epoch_logs.value)
else:
    exit()

plot_graph(epoch_logs)