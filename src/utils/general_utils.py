from typing import Any, Mapping
from dataclasses import dataclass, fields


class ChainableDict(dict):
    """Container object exposing keys as attributes.

    State objects extend dictionaries by enabling values to be accessed by key,
    `state["value_key"]`, or by an attribute, `state.value_key`.

    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        """Setattr method.

        Args:
            key: The input key.
            value: The corresponding value to the key.
        """
        self[key] = value

    def __dir__(self):
        """Method to return all the keys."""
        return self.keys()

    def __getattr__(self, key):
        """Method to access value associated with the key.

        Args:
            key: The input key.
        """
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError from exc


def dataclass_from_dict(cls: dataclass, src: Mapping[str, Any]) -> dataclass:
    """Create a new instance of the class from a dictionary.

    Reference: https://stackoverflow.com/questions/53376099/python-dataclass-from-a-nested-dict
    """
    field_types_lookup = {field.name: field.type for field in fields(cls)}

    constructor_inputs = {}
    for field_name, value in src.items():
        try:
            # recursive call to dataclass_from_dict
            constructor_inputs[field_name] = dataclass_from_dict(
                field_types_lookup[field_name], value
            )
        except TypeError:
            # type error from fields() call in recursive call
            # indicates that field is not a dataclass, this is how we are
            # breaking the recursion. If not a dataclass - no need for loading
            constructor_inputs[field_name] = value
        except KeyError:
            # similar, field not defined on dataclass, pass as plain field value
            constructor_inputs[field_name] = value
    return cls(**constructor_inputs)


if __name__ == "__main__":
    state1 = ChainableDict(a=1, b=2)
    state2 = ChainableDict(**{"a": 1, "b": 2})
    print(state2.a)
