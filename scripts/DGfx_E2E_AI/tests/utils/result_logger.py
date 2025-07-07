import copy
import json
from pathlib import Path

import yaml


def log_result(dic, log_path):
    # Create copy of results.
    result_dic = copy.deepcopy(dic)


    # Read deviceId.json from log_path
    deviceId_json = log_path / "deviceID.json"
    with open(deviceId_json, "r") as f:
        deviceId = json.load(f)

    # Read driverINFO.json from log_path
    driverINFO_json = log_path / "driverINFO.json"
    with open(driverINFO_json, "r") as f:
        driverINFO = json.load(f)
        
    # Read environment.json from log_path
    env_yml = log_path / "environment.yml"
    with open(env_yml, "r") as f:
        env_payload = yaml.safe_load(f)

    if deviceId:
        result_dic["deviceId"] = deviceId
    if driverINFO:
        result_dic["driverINFO"] = driverINFO
    if env_payload:
        result_dic["environment"] = env_payload

    result_dic['testResult'] = dic


    print('Overall Score: ', result_dic)
