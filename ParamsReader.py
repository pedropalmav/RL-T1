import json

class ParamsReader:
    @staticmethod
    def read_params(file_path: str) -> dict:
        with open(file_path) as json_file:
            return json.load(json_file)