import logging
import pickle
from pathlib import Path
from typing import Any, Iterable


class FileIO:
    @staticmethod
    def save(obj: object, path: Path, overwrite: bool = False):
        if path.exists() and not overwrite:
            logging.warning(f"{path} already exists. Skipping save.")
            return

        with open(path, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load(path: Path, overwrite: bool = False) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def save_it_to_csv(
        it: Iterable, path: Path, overwrite: bool = False, header: list = None
    ):
        if path.exists() and not overwrite:
            logging.warning(f"{path} already exists. Skipping save.")
            return

        with open(path, "w") as f:
            if header:
                f.write(f"{','.join(map(str, header))}\n")
            for item in it:
                f.write(f"{','.join(map(str, item[0].flatten()))},")
                f.write(f"{','.join(map(str, item[1].flatten()))}\n")
