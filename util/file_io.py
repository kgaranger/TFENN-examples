# Copyright 2023 Kévin Garanger
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
