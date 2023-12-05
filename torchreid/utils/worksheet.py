import json, os, gspread, requests, datetime, time, re
from oauth2client.service_account import ServiceAccountCredentials


KEY_FILE = "./excel-service-key.json"


def _get_worksheet(excel_link, worksheet_name):
    try:
        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        credentials = ServiceAccountCredentials.from_json_keyfile_name(KEY_FILE, scope)
        gc = gspread.authorize(credentials)

        document = gc.open_by_url(excel_link)
        worksheet = document.worksheet(worksheet_name)

        return worksheet
    except Exception as e:
        raise e


def _get_first_empty_row(worksheet):
    # Get all values in the first column (or any column you prefer)
    column_values = worksheet.col_values(1)  # Use the column number you want

    # Find the first empty cell
    for i, value in enumerate(column_values, start=1):
        if not value:
            return i

    # If all cells are occupied, return the next row
    return len(column_values) + 1

def _update_test_results_flow(worksheet, target_row, epochs_elapsed, test_results):
    cell_list = worksheet.range(f"F{target_row}:J{target_row}")

    worksheet_mAP_logs = worksheet.acell(f"F{target_row}")
    worksheet_rank1_logs = worksheet.acell(f"G{target_row}")
    worksheet_rank5_logs = worksheet.acell(f"H{target_row}")
    worksheet_rank10_logs = worksheet.acell(f"I{target_row}")
    worksheet_rank20_logs = worksheet.acell(f"J{target_row}")

    if isinstance(worksheet_mAP_logs.value, str):
        mAP_logs = json.loads(worksheet_mAP_logs.value)
    else:
        mAP_logs = {}
    if isinstance(worksheet_rank1_logs.value, str):
        rank1_logs = json.loads(worksheet_rank1_logs.value)
    else:
        rank1_logs = {}
    if isinstance(worksheet_rank5_logs.value, str):
        rank5_logs = json.loads(worksheet_rank5_logs.value)
    else:
        rank5_logs = {}
    if isinstance(worksheet_rank10_logs.value, str):
        rank10_logs = json.loads(worksheet_rank10_logs.value)
    else:
        rank10_logs = {}
    if isinstance(worksheet_rank20_logs.value, str):
        rank20_logs = json.loads(worksheet_rank20_logs.value)
    else:
        rank20_logs = {}

    mAP_logs.update({f"epoch_{epochs_elapsed}": test_results["mAP"]})
    rank1_logs.update({f"epoch_{epochs_elapsed}": test_results["Rank-1"]})
    rank5_logs.update({f"epoch_{epochs_elapsed}": test_results["Rank-5"]})
    rank10_logs.update({f"epoch_{epochs_elapsed}": test_results["Rank-10"]})
    rank20_logs.update({f"epoch_{epochs_elapsed}": test_results["Rank-20"]})

    cell_list[0].value = json.dumps(mAP_logs)
    cell_list[1].value = json.dumps(rank1_logs)
    cell_list[2].value = json.dumps(rank5_logs)
    cell_list[3].value = json.dumps(rank10_logs)
    cell_list[4].value = json.dumps(rank20_logs)

    worksheet.update_cells(cell_list)

def _update_test_results_analysis(worksheet, target_row, dataset_analyzed, model_on_analysis, test_results):

    # save model's name
    worksheet.update_acell(f"B{target_row}", model_on_analysis)

    # getting model short name
    pattern = re.compile(r'-r(\d+)-model\.pth\.tar-\d+')
    match = pattern.search(model_on_analysis)
    if match:
        short_name = match.group(1)
        worksheet.update_acell(f"C{target_row}", f"r{short_name}")


    if dataset_analyzed == 'market1501':
        cell_list = worksheet.range(f"D{target_row}:H{target_row}")
    elif dataset_analyzed == 'dukemtmcreid':
        cell_list = worksheet.range(f"I{target_row}:M{target_row}")

    if cell_list is not None:
        cell_list[0].value = test_results["mAP"]
        cell_list[1].value = test_results["Rank-1"]
        cell_list[2].value = test_results["Rank-5"]
        cell_list[3].value = test_results["Rank-10"]
        cell_list[4].value = test_results["Rank-20"]

        worksheet.update_cells(cell_list)

def update_worksheet(
    excel_link: str = None,
    worksheet_name: str = None,
    train_start_time: float = None,
    train_time_elapsed: float = None,
    comments: str = None,
    epochs_elapsed: int = None,
    last_epoch_summary: dict = None,
    test_results: dict = None,
    weights_produced: str = None,
    metadata: dict = None,
    is_analysis: bool = False,
    model_on_analysis: str = None,
    dataset_analyzed: str = None,
    session_completed: bool = False,
):
    try:
        worksheet_connector_path = os.environ.get('WORKSHEET_CONNECTOR_PATH')
        if excel_link is not None and worksheet_name is not None:
            worksheet = _get_worksheet(excel_link, worksheet_name)
            target_row = _get_first_empty_row(worksheet)

            connector_data = {
                        "excel_link": excel_link,
                        "worksheet_name": worksheet_name,
                        "target_row": target_row,
                        'is_analysis': is_analysis
                    }
            if is_analysis:
                connector_data.update({'model_on_analysis': model_on_analysis})
           
            with open(worksheet_connector_path, "w") as json_file:
                json.dump(
                    connector_data,
                    json_file,
                    indent=4,
                )
                json_file.close()

        if os.path.exists(worksheet_connector_path):
            json_data = {}
            with open(worksheet_connector_path, "r") as json_file:
                json_data = json.load(json_file)
                json_file.close()

            excel_link = json_data["excel_link"]
            worksheet_name = json_data["worksheet_name"]
            target_row = json_data["target_row"]
            is_analysis = json_data["is_analysis"]
            model_on_analysis = json_data['model_on_analysis'] if is_analysis else None
            worksheet = _get_worksheet(excel_link, worksheet_name)

            # Add all conditional cell values =====================

            if comments is not None:
                worksheet.update_acell(f"A{target_row}", comments)

            if train_start_time is not None:
                train_start_time = datetime.datetime.fromtimestamp(train_start_time)
                train_start_time = train_start_time.strftime("%Y-%m-%d %H:%M:%S")
                worksheet.update_acell(f"B{target_row}", train_start_time)
                worksheet.update_acell(f"D{target_row}", 0)

            if train_time_elapsed is not None and epochs_elapsed is not None and last_epoch_summary is not None:
                cell_list = worksheet.range(f"C{target_row}:E{target_row}")
                summary_obj = {
                    "loss": f"{last_epoch_summary['loss']:.4f}",
                    "acc": f"{last_epoch_summary['acc']:.4f}",
                }
                worksheet_epoch_logs = worksheet.acell(f"E{target_row}")
                if isinstance(worksheet_epoch_logs.value, str):
                    epoch_logs = json.loads(worksheet_epoch_logs.value)
                else:
                    epoch_logs = {}

                epoch_logs.update({f"epoch_{epochs_elapsed}": summary_obj})
                
                cell_list[0].value = train_time_elapsed
                cell_list[1].value = epochs_elapsed
                cell_list[2].value = json.dumps(epoch_logs)

                worksheet.update_cells(cell_list)
            
            elif train_time_elapsed is not None:
                worksheet.update_acell(f"C{target_row}", train_time_elapsed)

            if is_analysis == False and epochs_elapsed is not None and test_results is not None:
                _update_test_results_flow(worksheet, target_row, epochs_elapsed, test_results)
            elif is_analysis == True and test_results is not None:
                _update_test_results_analysis(worksheet, target_row, dataset_analyzed, model_on_analysis, test_results)

            if weights_produced is not None:
                worksheet.update_acell(f"K{target_row}", weights_produced)

            if metadata is not None:
                sheet_meta = worksheet.acell(f"L{target_row}")
                if isinstance(sheet_meta.value, str):
                    sheet_meta = json.loads(sheet_meta.value)
                else:
                    sheet_meta = {}
                sheet_meta.update(metadata)
                worksheet.update_acell(f"L{target_row}", json.dumps(sheet_meta))

            # set updated time
            updated_at = datetime.datetime.fromtimestamp(time.time())
            updated_at = updated_at.strftime("%Y-%m-%d %H:%M:%S")
            update_column = 'M' if is_analysis == False else 'A'
            worksheet.update_acell(f"{update_column}{target_row}", updated_at)
            
            if session_completed == True:
                os.remove(worksheet_connector_path)

        else:
            return

    except Exception as e:
        print("Error " + str(e))
