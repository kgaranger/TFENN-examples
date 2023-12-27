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
