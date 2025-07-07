import json
import os
from pathlib import Path

ADAPTER_INFO_FILE_PATH = "C:\\Automation_Tools\\adapter_details.json"
if not os.path.exists(ADAPTER_INFO_FILE_PATH):
    ADAPTER_INFO_FILE_PATH = "C:\\e2e_auto_val\\taskml_details.json"

def dir_traversal_search(dir_name):
    traverse_count = 0
    current_path = Path.cwd()  # Get the current working directory
    # Iterate through parent directories until reaching the root directory ('/')
    while current_path != current_path.parent:  # Check if current_path is not the root
        if current_path.name == dir_name:
            return True, traverse_count, current_path  # Found the directory
        current_path = current_path.parent  # Move to the parent directory
        traverse_count += 1
    return False, 0, current_path
    
def file_traversal_search(file_name):
    traverse_count = 0
    current_path = Path.cwd()  # Get the current working directory
    # Iterate through parent directories until reaching the root directory ('/')
    listdir = [i for i in current_path.glob(file_name)]
    while current_path != current_path.parent:  # Check if current_path is not the root
        if len(listdir) != 0:
            return True, traverse_count, listdir[0]  # Found the directory
        current_path = current_path.parent  # Move to the parent directory
        listdir = [i for i in current_path.glob(file_name)]
        traverse_count += 1
    return False, 0, listdir
    
def get_gtax_test_dir():
    _, _, test_path = file_traversal_search(".traversal_search_aiml")
    return test_path.parent

def update_log_details():
    # Add Log info into json file for hook to pull on timeout
    test_path = get_gtax_test_dir()
    log_path = str(test_path / "logs")
    
    taskml_details_path = Path(ADAPTER_INFO_FILE_PATH)
    log_info = {"logs_dir": log_path}

    # if exists (100% for automation env using test_e2e plugin)
    if taskml_details_path.exists():
        # Adding this default addition for local runs
        if taskml_details_path.stat().st_size > 0:
            # file is not empty
            print(f"{taskml_details_path} file exists. Reading contents to " "add logging details")
            with open(str(taskml_details_path), "r") as file:
                data = json.load(file)
        else:
            # file is empty, writing default values.
            print(f"{taskml_details_path} file does not exist. Creating new file with " "default contents")
            data = {"adapter_info": {"render": "IGfx", "display_out": "IGfx"}}
        if "log_details" in data.keys():
            print("Log details Already found in json, skipping")
        else:
            print("Updating Log details Assuming Script run from AIML")
            data.update({"log_details": log_info})
        with open(str(taskml_details_path), "w") as file:
            json.dump(data, file, indent=4, separators=(",", ": "))
        print(f"{taskml_details_path} update with logging details")
    else:
        print(f"{taskml_details_path} does not exist. Execution not running via test_e2e plugin. Resuming.")