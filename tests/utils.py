from numbers import Number


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
