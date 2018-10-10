from numbers import Number


def parse_key(key):
    for constructor in [int, float, str]:
        try:
            return constructor(key)
        except ValueError:
            pass
    raise ValueError("Key could not be parsed into int, float or str")


def convert_json(obj, parse_strings=True):
    if isinstance(obj, (Number, bool, type(None))):
        new = obj
    elif isinstance(obj, str):
        if parse_strings:
            new = parse_key(obj)
        else:
            new = obj
    elif isinstance(obj, list):
        new = [convert_json(item) for item in obj]
    elif isinstance(obj, dict):
        new = {parse_key(key): convert_json(value) for key, value in obj.items()}
    else:
        raise TypeError(
            f"Given object was of type {type(obj).__name__}, not str/int/float/bool/NoneType/list/dict"
        )

    return new
