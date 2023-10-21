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

def plot_graph(data):
        # Extract loss and accuracy values
    epochs = [int(epoch.split("_")[1]) for epoch in data.keys()]
    loss_values = [float(data[epoch]["loss"]) for epoch in data.keys()]
    acc_values = [float(data[epoch]["acc"]) for epoch in data.keys()]

    # Define custom colors for the plots
    loss_color = "red"
    acc_color = "green"

    # Create separate figures
    fig_loss = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Loss Over Epochs", "Accuracy Over Epochs"),
    )

    # Add Loss plot
    fig_loss.add_trace(
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
    fig_loss.add_trace(
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

    # Update layout for each figure
    fig_loss.update_xaxes(title_text="Epoch", row=2, col=1)
    fig_loss.update_xaxes(title_text="Epoch", row=1, col=1)
    fig_loss.update_yaxes(title_text="Loss", row=1, col=1)
    fig_loss.update_yaxes(title_text="Accuracy (%)", row=2, col=1)

    # Display both figures together
    fig_loss.show()

WORKSHEET_NAME = "Finetune without RP"
TARGET_ROW = 8

worksheet = _get_worksheet()
# @TODO: After removing last_epoch_summary (E Column) from sheets, update the targeted cell's F with E.
epoch_logs = worksheet.acell(f'F{TARGET_ROW}')

if isinstance(epoch_logs.value, str):
    epoch_logs = json.loads(epoch_logs.value)
else:
    exit()

plot_graph(epoch_logs)