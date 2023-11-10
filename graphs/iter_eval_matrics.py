import json, os, gspread, requests, datetime, time
from oauth2client.service_account import ServiceAccountCredentials
import plotly.graph_objects as go

# Your matrices and corresponding data
sample_data = {
    "mAP": {"row_5": "9.723%", "row_6": "9.828%", "row_8": "11.215%"},
    "Rank-1": {"row_5": "12.544%", "row_6": "12.544%", "row_8": "13.365%"},
    "Rank-5": {"row_5": "14.537%", "row_6": "14.185%", "row_8": "15.123%"},
    "Rank-10": {"row_5": "15.592%", "row_6": "14.889%", "row_8": "15.240%"},
    "Rank-20": {"row_5": "17.116%", "row_6": "15.826%", "row_8": "16.295%"},
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
        xaxis_title="Iterations (rows)",
        yaxis_title="Percentage %",
    )
    _save_graph(fig)


WORKSHEET_NAME = "Finetune with RP"
START_ROW = 2
END_ROW = 15

worksheet = _get_worksheet()
mAP_logs = worksheet.col_values(6)[START_ROW - 1 : END_ROW]
rank1_logs = worksheet.col_values(7)[START_ROW - 1 : END_ROW]
rank5_logs = worksheet.col_values(8)[START_ROW - 1 : END_ROW]
rank10_logs = worksheet.col_values(9)[START_ROW - 1 : END_ROW]
rank20_logs = worksheet.col_values(10)[START_ROW - 1 : END_ROW]

eval_data = {"mAP": {}, "Rank-1": {}, "Rank-5": {}, "Rank-10": {}, "Rank-20": {}}

for index in range(len(mAP_logs)):
    row = index + START_ROW

    mAP_data = mAP_logs[index]
    if len(mAP_data) > 0:
        mAP_data = json.loads(mAP_data)
        last_epoch = list(mAP_data.keys())[-1]
        eval_data["mAP"][f"row_{row}"] = mAP_data[last_epoch]

    rank1_data = rank1_logs[index]
    if len(rank1_data) > 0:
        rank1_data = json.loads(rank1_data)
        last_epoch = list(rank1_data.keys())[-1]
        eval_data["Rank-1"][f"row_{row}"] = rank1_data[last_epoch]

    rank5_data = rank5_logs[index]
    if len(rank5_data) > 0:
        rank5_data = json.loads(rank5_data)
        last_epoch = list(rank5_data.keys())[-1]
        eval_data["Rank-5"][f"row_{row}"] = rank5_data[last_epoch]

    rank10_data = rank10_logs[index]
    if len(rank10_data) > 0:
        rank10_data = json.loads(rank10_data)
        last_epoch = list(rank10_data.keys())[-1]
        eval_data["Rank-10"][f"row_{row}"] = rank10_data[last_epoch]

    rank20_data = rank20_logs[index]
    if len(rank20_data) > 0:
        rank20_data = json.loads(rank20_data)
        last_epoch = list(rank20_data.keys())[-1]
        eval_data["Rank-20"][f"row_{row}"] = rank20_data[last_epoch]

eval_data = {key: value for key, value in eval_data.items() if value}

if len(eval_data.keys()) > 0:
    plot_graph(eval_data)
