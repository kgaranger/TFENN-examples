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

from typing import Any


def key_value_pair(arg: Any) -> tuple[bool, tuple[str, str] | None]:
    """Helper function for sub_argparse to parse key-value pairs"""
    if "=" in arg:
        key, value = arg.split("=")
        return True, (key, value)
    else:
        return False, None


def list_to_args(args: list[Any]) -> tuple[list[Any], dict[str, Any]]:
    """Helper function for sub_argparse to parse key-value pairs"""
    new_args = []
    kwargs = {}
    for arg in args:
        is_key_value_pair, key_value = key_value_pair(arg)
        if is_key_value_pair:
            key, value = key_value
            kwargs[key] = value
        else:
            if len(kwargs) > 0:
                raise ValueError(
                    "Positional arguments must come before keyword arguments"
                )
            new_args.append(arg)
    return new_args, kwargs
