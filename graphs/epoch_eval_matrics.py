import sys, re
sys.path.append("/home/code/Shahzaib/MS/Thesis/Implementation/deep-person-reid/")
from helpers import Matric

import json, os, gspread, requests, datetime, time
from oauth2client.service_account import ServiceAccountCredentials
import plotly.graph_objects as go

# Your matrices and corresponding data
sample_data = {
    'mAP': {"epoch_10": "9.723%", "epoch_20": "9.828%", "epoch_30": "11.215%"},
    'Rank-1': {"epoch_10": "12.544%", "epoch_20": "12.544%", "epoch_30": "13.365%"},
    'Rank-5': {"epoch_10": "14.537%", "epoch_20": "14.185%", "epoch_30": "15.123%"},
    'Rank-10': {"epoch_10": "15.592%", "epoch_20": "14.889%", "epoch_30": "15.240%"},
    'Rank-20': {"epoch_10": "17.116%", "epoch_20": "15.826%", "epoch_30": "16.295%"}
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
    file_path = os.path.join(directory, f'r{TARGET_ROW}-epoch-eval.html')
    fig.write_html(file_path, auto_open=True)

def plot_graph(data): 
    # Function to convert percentage string to float
    def convert_percentage_to_float(percentage_str):
        return float(percentage_str.strip('%'))

    # Extracting epochs
    epochs = list(next(iter(data.values())).keys())

    # Creating the plot
    fig = go.Figure()

    for metric, values in data.items():
        fig.add_trace(go.Scatter(
            x=epochs,
            y=[convert_percentage_to_float(values[epoch]) for epoch in epochs],
            mode='lines+markers',
            name=metric
        ))

    fig.update_traces(hoverlabel=dict(font_color='white'))
    fig.update_layout(title="Evaluation Metrics over Epochs",
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
        ),
    xaxis_title="Epochs",
    yaxis_title="Percentage %"
    )
    _save_graph(fig)

WORKSHEET_NAME = "Finetune with RP - ResNet50"
TARGET_ROW = 11

worksheet = _get_worksheet()
mAP_logs = worksheet.acell(f'F{TARGET_ROW}')
rank1_logs = worksheet.acell(f'G{TARGET_ROW}')
rank5_logs = worksheet.acell(f'H{TARGET_ROW}')
rank10_logs = worksheet.acell(f'I{TARGET_ROW}')
rank20_logs = worksheet.acell(f'J{TARGET_ROW}')

eval_data = {}

if isinstance(mAP_logs.value, str):
    eval_data[Matric.map] = json.loads(mAP_logs.value)
if isinstance(rank1_logs.value, str):
    eval_data[Matric.rank1] = json.loads(rank1_logs.value)
if isinstance(rank5_logs.value, str):
    eval_data[Matric.rank5] = json.loads(rank5_logs.value)
if isinstance(rank10_logs.value, str):
    eval_data[Matric.rank10] = json.loads(rank10_logs.value)
if isinstance(rank20_logs.value, str):
    eval_data[Matric.rank20] = json.loads(rank20_logs.value)

if len(eval_data.keys()) > 0:
    plot_graph(eval_data)