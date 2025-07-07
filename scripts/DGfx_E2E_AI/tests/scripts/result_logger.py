import copy
import json
from pathlib import Path


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

    if deviceId:
        result_dic["deviceId"] = deviceId
    if driverINFO:
        result_dic["driverINFO"] = driverINFO

    environment = _convert_env_to_json(log_path)
    result_dic['environment'] = environment

    result_dic['testResult'] = dic


    print('Overall Score: ', result_dic)


def _convert_env_to_json(log_path):
    try:
        env_payload = Path(log_path) /"environment.txt"
        with open(env_payload, "r") as yaml_in:
            return json.loads(yaml_in.read())
    except Exception as e:
        return Exception("Failed to read environment file", e)
