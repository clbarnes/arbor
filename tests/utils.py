import os
import json
from numbers import Number

from tests.constants import TEST_SKELETON, DATA_ROOT


def parse_key(key):
    for constructor in [int, float, str]:
        try:
            return constructor(key)
        except ValueError:
            pass
    raise ValueError("Key could not be parsed into int, float or str")


def from_jso(obj, parse_strings=True):
    """
    Recursively convert object from what would be returned by ``json.load``.

    Specifically, replace string-serialied numeric ``dict`` keys with numbers.

    If ``parse_strings`` is ``True``, string-serialised numeric ``dict`` values and ``list`` items will also
    be converted into numbers.
    """
    if isinstance(obj, (Number, bool, type(None))):
        new = obj
    elif isinstance(obj, str):
        if parse_strings:
            new = parse_key(obj)
        else:
            new = obj
    elif isinstance(obj, list):
        new = [from_jso(item) for item in obj]
    elif isinstance(obj, dict):
        new = {parse_key(key): from_jso(value) for key, value in obj.items()}
    else:
        raise TypeError(
            f"Given object was of type {type(obj).__name__}, not str/int/float/bool/NoneType/list/dict"
        )

    return new


def to_jso_like(obj):
    """
    Recursively convert object into something which can be ``json.dump``ed.

    Specifically, replace tuples with lists, sets with sorted lists, and call the ``to_dict``
    method of anything which has it.

    Output is not quite a JSO because dicts can have numeric keys.
    """
    if isinstance(obj, (str, Number, bool, type(None))):
        new = obj
    elif isinstance(obj, set):
        new = to_jso_like(sorted(obj))
    elif isinstance(obj, (list, tuple)):
        new = [to_jso_like(item) for item in obj]
    elif isinstance(obj, dict):
        new = {to_jso_like(key): to_jso_like(value) for key, value in obj.items()}
    elif hasattr(obj, "to_dict"):
        new = to_jso_like(obj.to_dict())
    else:
        raise TypeError(
            f"Given object was of type {type(obj).__name__}, not str/int/float/bool/NoneType/list/dict"
        )

    return new


def get_expected(path_item, *path_items):
    path_items = [path_item] + list(path_items)
    path_items[-1] += ".json"
    return load_json(str(TEST_SKELETON), "reference", *path_items)


def fixture_path(path_item, *path_items):
    return os.path.join(DATA_ROOT, path_item, *path_items)


def load_json(path_item, *path_items, parse_keys=True, parse_strings=True):
    """Get JSON data from fixture file"""
    with open(fixture_path(path_item, *path_items)) as f:
        obj = json.load(f)

    if parse_keys:
        return from_jso(obj, parse_strings=parse_strings)
    else:
        return obj
