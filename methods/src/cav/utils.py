import json


def difference(a: dict, b: dict) -> dict:
    result = {}
    for k in a.keys():
        result[k] = {j: a[k][j] - b[k][j] for j in a[k].keys()}
    return result


def log_data(
    data: dict, folder: str = "methods/src/cav/log", filename: str = "log"
) -> None:
    with open(f"{folder}/{filename}.json", "w") as f:
        json.dump(data, f, indent=4)
