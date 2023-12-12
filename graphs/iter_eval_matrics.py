import sys, re
sys.path.append("/home/code/Shahzaib/MS/Thesis/Implementation/deep-person-reid/")
from helpers import Matric
import json, os, gspread, requests, datetime, time
from oauth2client.service_account import ServiceAccountCredentials
import plotly.graph_objects as go

# Your matrices and corresponding data
sample_data = {
    "mAP": {"r5": "9.723%", "r6": "9.828%", "r8": "11.215%"},
    "Rank-1": {"r5": "12.544%", "r6": "12.544%", "r8": "13.365%"},
    "Rank-5": {"r5": "14.537%", "r6": "14.185%", "r8": "15.123%"},
    "Rank-10": {"r5": "15.592%", "r6": "14.889%", "r8": "15.240%"},
    "Rank-20": {"r5": "17.116%", "r6": "15.826%", "r8": "16.295%"},
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
        print("exception: ", e)

def _save_graph(fig): 
    # Save the figure as an HTML file and open in browser
    directory = f"./training_results/{WORKSHEET_NAME}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f'r{START_ROW}-r{END_ROW}-iter-eval.html')
    fig.write_html(file_path, auto_open=True)


def plot_graph(data):
    # Function to convert percentage string to float
    def convert_percentage_to_float(percentage_str):
        return float(percentage_str.strip("%"))

    # Extracting epochs
    epochs = list(next(iter(data.values())).keys())

    # Creating the plot
    fig = go.Figure()

    for metric, values in data.items():
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[convert_percentage_to_float(values[epoch]) for epoch in epochs],
                mode="lines+markers",
                name=metric,
            )
        )

    fig.update_traces(hoverlabel=dict(font_color='white'))
    fig.update_layout(
        title="Evaluation Metrics over Iterations",
        plot_bgcolor='#f7f7f7',
        xaxis_title="Iterations (rows)",
        yaxis_title="Percentage %",
        annotations=[
            dict(
                text=f"{WORKSHEET_NAME} - R{START_ROW} to R{END_ROW}",
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


WORKSHEET_NAME = "Finetune with RP - ResNet50"
START_ROW = 5
END_ROW = 8

worksheet = _get_worksheet()
mAP_logs = worksheet.col_values(6)[START_ROW - 1 : END_ROW]
rank1_logs = worksheet.col_values(7)[START_ROW - 1 : END_ROW]
rank5_logs = worksheet.col_values(8)[START_ROW - 1 : END_ROW]
rank10_logs = worksheet.col_values(9)[START_ROW - 1 : END_ROW]
rank20_logs = worksheet.col_values(10)[START_ROW - 1 : END_ROW]

eval_data = {Matric.map: {}, Matric.rank1: {}, Matric.rank5: {}, Matric.rank10: {}, Matric.rank20: {}}

for index in range(len(mAP_logs)):
    row = index + START_ROW

    mAP_data = mAP_logs[index]
    if len(mAP_data) > 0:
        mAP_data = json.loads(mAP_data)
        last_epoch = list(mAP_data.keys())[-1]
        eval_data["mAP"][f"r{row}"] = mAP_data[last_epoch]

    rank1_data = rank1_logs[index]
    if len(rank1_data) > 0:
        rank1_data = json.loads(rank1_data)
        last_epoch = list(rank1_data.keys())[-1]
        eval_data["Rank-1"][f"r{row}"] = rank1_data[last_epoch]

    rank5_data = rank5_logs[index]
    if len(rank5_data) > 0:
        rank5_data = json.loads(rank5_data)
        last_epoch = list(rank5_data.keys())[-1]
        eval_data["Rank-5"][f"r{row}"] = rank5_data[last_epoch]

    rank10_data = rank10_logs[index]
    if len(rank10_data) > 0:
        rank10_data = json.loads(rank10_data)
        last_epoch = list(rank10_data.keys())[-1]
        eval_data["Rank-10"][f"r{row}"] = rank10_data[last_epoch]

    rank20_data = rank20_logs[index]
    if len(rank20_data) > 0:
        rank20_data = json.loads(rank20_data)
        last_epoch = list(rank20_data.keys())[-1]
        eval_data["Rank-20"][f"r{row}"] = rank20_data[last_epoch]

eval_data = {key: value for key, value in eval_data.items() if value}

if len(eval_data.keys()) > 0:
    plot_graph(eval_data)
