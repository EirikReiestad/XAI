import os


class FileHandler:
    @staticmethod
    def file_exist(dir_path: str, filename: str):
        """Checks if the file exists."""
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory {dir_path} does not exist.")
        if not os.path.isfile(dir_path + filename):
            raise FileNotFoundError(f"File {filename} does not exist.")
