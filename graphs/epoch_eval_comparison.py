import sys
sys.path.append("/home/code/Shahzaib/MS/Thesis/Implementation/deep-person-reid/")
from helpers import Matric

import json, os, gspread, requests, datetime, time
from oauth2client.service_account import ServiceAccountCredentials
import plotly.graph_objects as go
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

    def make_file_name():
        sorted_list = sorted(TARGET_ROWS)
        file_name = "-".join([f"r{row}" for row in sorted_list] + ["epoch", "eval", "comp"])
        return file_name

    # Save the figure as a single HTML file and open in browser
    directory = f"./training_results/{WORKSHEET_NAME}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f'{make_file_name()}.html')
    fig.write_html(file_path, auto_open=True)

def plot_graph(all_data, matrices): 
   # Function to convert percentage string to float
    def convert_percentage_to_float(percentage_str):
        return float(percentage_str.strip('%'))

    # Creating the plot
    fig = go.Figure()

    for row, data in all_data.items():
        for metric in matrices:
            metric_data = data.get(metric)
            if metric_data:
                # Extracting epochs and corresponding values
                epochs = list(metric_data.keys())
                values = [convert_percentage_to_float(metric_data[epoch]) for epoch in epochs]

                fig.add_trace(go.Scatter(
                    x=epochs,
                    y=values,
                    mode='lines+markers',
                    name=f"{metric} - r{row}"
                ))

    def make_title(WORKSHEET_NAME, rows):
        # Sort the list
        sorted_list = sorted(rows)

        # Create the text
        title = f"{WORKSHEET_NAME} - r{', r'.join(map(str, sorted_list))}"
        return title

    # Update layout
    fig.update_layout(
        title="Matrices Evaluation Comparison over Epochs",
        plot_bgcolor='#f7f7f7',
        xaxis_title="Epochs",
        yaxis_title="Percentage %",
        annotations=[
            dict(
                text=make_title(WORKSHEET_NAME, TARGET_ROWS),
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

    # Save the figure
    _save_graph(fig)

WORKSHEET_NAME = "Finetune with RP - ResNet50"
TARGET_ROWS = [ 8, 10] 
MATRICES = [Matric.rank5, Matric.map]

worksheet = _get_worksheet()
all_eval_data = {}

for TARGET_ROW in TARGET_ROWS:
    row_data = {
        Matric.map: worksheet.acell(f'F{TARGET_ROW}').value,
        Matric.rank1: worksheet.acell(f'G{TARGET_ROW}').value,
        Matric.rank5: worksheet.acell(f'H{TARGET_ROW}').value,
        Matric.rank10: worksheet.acell(f'I{TARGET_ROW}').value,
        Matric.rank20: worksheet.acell(f'J{TARGET_ROW}').value
    }

    eval_data = {metric: json.loads(value) for metric, value in row_data.items() if value and metric in MATRICES}
    if eval_data:
        all_eval_data[TARGET_ROW] = eval_data

if all_eval_data:
    plot_graph(all_eval_data, MATRICES)