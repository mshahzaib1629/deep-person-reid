import sys, re
sys.path.append("/home/code/Shahzaib/MS/Thesis/Implementation/deep-person-reid/")
import pandas as pd
import plotly.graph_objects as go
import json, os, gspread, requests, datetime, time
from oauth2client.service_account import ServiceAccountCredentials
from helpers import Matric, SelectedDatasets

# The data dictionary as provided
# data = {
#     "models_trained": ["r12", "r24", "r25"],
#     "market1501": {
#         "mAP": ["39.48%", "44.6%", "38.9%"],
#         "Rank-1": ["66.1%", "68.5%", "63.4%"],
#         "Rank-5": ["83.9%", "84.6%", "81.7%"]
#     },
#     "dukemtmcreid": {
#         "mAP": ["10.9%", "24.1%", "23.3%"],
#         "Rank-1": ["22.2%", "43.9%", "45.01%"],
#         "Rank-5": ["36.7%", "61.1%", "63.8%"]
#     }
# }

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

def _get_data(rows, datasets, matrices):
    worksheet = _get_worksheet()
    data = {}

    def add_data (dataset, matric, row, column):
        cell_value = worksheet.acell(f'{column}{row}').value
        if cell_value != None:
            data[dataset][matric].append(cell_value)

    for row in rows:
        if 'models_trained' not in data:
             data['models_trained'] = []
        model_short_name = worksheet.acell(f'C{row}').value
        if model_short_name != None:
            data["models_trained"].append(model_short_name)

        for dataset in datasets:
            # create dataset structure in data if its not present
            if dataset not in data.keys():
                data[dataset] = {matric: [] for matric in matrices}
            
            if dataset == SelectedDatasets.Market1501:
                    add_data(dataset, Matric.map, row, 'D') if Matric.map in matrices else None
                    add_data(dataset, Matric.rank1, row, 'E') if Matric.rank1 in matrices else None
                    add_data(dataset, Matric.rank5, row, 'F') if Matric.rank5 in matrices else None
                    add_data(dataset, Matric.rank10, row, 'G') if Matric.rank10 in matrices else None
                    add_data(dataset, Matric.rank20, row, 'H') if Matric.rank20 in matrices else None

            elif dataset == SelectedDatasets.DukeMTMC:
                    add_data(dataset, Matric.map, row, 'I') if Matric.map in matrices else None
                    add_data(dataset, Matric.rank1, row, 'J') if Matric.rank1 in matrices else None
                    add_data(dataset, Matric.rank5, row, 'K') if Matric.rank5 in matrices else None
                    add_data(dataset, Matric.rank10, row, 'L') if Matric.rank10 in matrices else None
                    add_data(dataset, Matric.rank20, row, 'M') if Matric.rank20 in matrices else None
    # print(data)
    return data

def _save_graph(data, fig): 
    
    def make_file_name():
        models_sorted = sorted(data['models_trained'])
        datasets_sorted_list = sorted(DATASETS)
        matrices_sorted_list = sorted(MATRICES)

        file_name = "-".join([f"{model}" for model in models_sorted]) + "_" + "-".join([f"{dataset}" for dataset in datasets_sorted_list]) + "_" + "-".join(f"{matric}" for matric in matrices_sorted_list)
        return file_name
    
    # worksheet name without Analysis keyword
    main_worksheet_name = re.sub(r'\[.*?\]', '', WORKSHEET_NAME).strip()
    directory =  os.path.join(".", "training_results", main_worksheet_name)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    analysis_path = os.path.join(directory, 'analysis')
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)
    
    file_path = os.path.join(analysis_path, f'{make_file_name()}.html')
    fig.write_html(file_path, auto_open=True)

def plot_graph(data, rows, metrics, datasets):
    
    def convert_percentage_to_float(percentage_str):
        return float(percentage_str.strip('%'))

    # Filter rows (subtract 1 for zero-indexed DataFrame)
    rows = [index for index, value in enumerate(data["models_trained"])]
    # Initialize a figure
    fig = go.Figure()

    # Create traces for each selected metric
    for metric in metrics:
        for dataset in datasets:
            # Prepare the metric data for the selected rows
            metric_data = [convert_percentage_to_float(data[dataset][metric][r]) for r in rows]
            # Add a trace for this metric
            fig.add_trace(go.Scatter(
                x=[data["models_trained"][r] for r in rows],
                y=metric_data,
                mode='lines+markers',
                name=f'{dataset} {metric}'
            ))

    def make_title(WORKSHEET_NAME, rows):
        models_sorted = sorted(data['models_trained'])
        datasets_sorted_list = sorted(DATASETS)
        matrices_sorted_list = sorted(MATRICES)
        # Sort the list
        sorted_list = sorted(rows)

        # Create the text
        title = f"{WORKSHEET_NAME} - {', '.join(map(str, models_sorted))} - {', '.join(map(str, datasets_sorted_list))} - {', '.join(map(str, matrices_sorted_list))} "
        return title
    
    # Update the layout of the figure
    fig.update_layout(
        title='Trained Models Perfornmace on Datasets',
        plot_bgcolor='#f7f7f7',
        xaxis_title='Models Trained',
        yaxis_title='Metric Values',
        legend_title='Metrics',
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
        )],
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

    _save_graph(data, fig)
 

WORKSHEET_NAME = "[Analysis] Finetune with RP - ResNet50"
TARGET_ROWS = [3, 4, 5, 6, 7]
MATRICES = [ Matric.rank5, ]
DATASETS = [SelectedDatasets.Market1501, SelectedDatasets.DukeMTMC]

data = _get_data(TARGET_ROWS, DATASETS, MATRICES)
plot_graph(data, TARGET_ROWS, MATRICES, DATASETS)
