import json, os, gspread, requests, datetime, time
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
    session_completed: bool = False,
):
    try:
        file_path = "./temp.json"

        if excel_link is not None and worksheet_name is not None:
            worksheet = _get_worksheet(excel_link, worksheet_name)
            target_row = _get_first_empty_row(worksheet)

            with open(file_path, "w") as json_file:
                json.dump(
                    {
                        "excel_link": excel_link,
                        "worksheet_name": worksheet_name,
                        "target_row": target_row,
                    },
                    json_file,
                    indent=4,
                )
                json_file.close()

        if os.path.exists(file_path):
            json_data = {}
            with open(file_path, "r") as json_file:
                json_data = json.load(json_file)
                json_file.close()

            excel_link = json_data["excel_link"]
            worksheet_name = json_data["worksheet_name"]
            target_row = json_data["target_row"]

            worksheet = _get_worksheet(excel_link, worksheet_name)

            # Add all conditional cell values =====================

            if comments is not None:
                worksheet.update_acell(f"A{target_row}", comments)

            if train_start_time is not None:
                train_start_time = datetime.datetime.fromtimestamp(train_start_time)
                train_start_time = train_start_time.strftime("%Y-%m-%d %H:%M:%S")
                worksheet.update_acell(f"B{target_row}", train_start_time)
                worksheet.update_acell(f"D{target_row}", 0)

            if train_time_elapsed is not None:
                worksheet.update_acell(f"C{target_row}", train_time_elapsed)

            if epochs_elapsed is not None and last_epoch_summary is not None:
                cell_list = worksheet.range(f"D{target_row}:F{target_row}")
                summary_obj = {
                    "loss": f"{last_epoch_summary['loss']:.4f}",
                    "acc": f"{last_epoch_summary['acc']:.4f}",
                }
                worksheet_epoch_logs = worksheet.acell(f"F{target_row}")
                if isinstance(worksheet_epoch_logs.value, str):
                    epoch_logs = json.loads(worksheet_epoch_logs.value)
                else:
                    epoch_logs = {}

                epoch_logs.update({f"epoch_{epochs_elapsed}": summary_obj})
                cell_list[0].value = epochs_elapsed
                cell_list[1].value = json.dumps(summary_obj)
                cell_list[2].value = json.dumps(epoch_logs)

                worksheet.update_cells(cell_list)

            if test_results is not None:
                cell_list = worksheet.range(f"G{target_row}:K{target_row}")
                cell_list[0].value = test_results["mAP"]
                cell_list[1].value = test_results["Rank-1"]
                cell_list[2].value = test_results["Rank-5"]
                cell_list[3].value = test_results["Rank-10"]
                cell_list[4].value = test_results["Rank-20"]

                worksheet.update_cells(cell_list)

            if weights_produced is not None:
                worksheet.update_acell(f"L{target_row}", weights_produced)

            if metadata is not None:
                sheet_meta = worksheet.acell(f"M{target_row}")
                if isinstance(sheet_meta.value, str):
                    sheet_meta = json.loads(sheet_meta.value)
                else:
                    sheet_meta = {}
                sheet_meta.update(metadata)
                worksheet.update_acell(f"M{target_row}", json.dumps(sheet_meta))

            if session_completed == True:
                session_completed_at = datetime.datetime.fromtimestamp(time.time())
                session_completed_at = session_completed_at.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                worksheet.update_acell(f"N{target_row}", session_completed_at)

                os.remove(file_path)

        else:
            return

    except Exception as e:
        print("Error " + str(e))
