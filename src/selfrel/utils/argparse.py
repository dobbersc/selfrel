from argparse import Action, ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace, RawTextHelpFormatter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Optional, TypeVar, Union

__all__ = ["RawTextArgumentDefaultsHelpFormatter", "StoreDictKeyPair", "none_or_str", "none_or_path"]


T = TypeVar("T")


class RawTextArgumentDefaultsHelpFormatter(RawTextHelpFormatter, ArgumentDefaultsHelpFormatter):
    pass


class StoreDictKeyPair(Action):
    # noinspection PyShadowingBuiltins
    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        nargs: Union[int, str, None] = None,
        const: Optional[T] = None,
        default: Union[T, str, None] = None,
        key_type: Optional[Callable[[str], Any]] = None,
        value_type: Optional[Callable[[str], Any]] = None,
        required: bool = False,
        help: Optional[str] = None,  # noqa: A002
        metavar: Union[str, tuple[str, ...], None] = None,
    ) -> None:
        if nargs is None:
            msg = f"{type(self).__name__} requires 'nargs' to be set unequal to 'None'."
            raise ValueError(msg)

        super().__init__(
            option_strings,
            dest,
            nargs=nargs,
            const=const,
            default=default,
            type=None,  # str
            required=required,
            help=help,
            metavar=metavar,
        )
        self.key_type = key_type
        self.value_type = value_type

    def __call__(
        self,
        parser: ArgumentParser,  # noqa: ARG002
        namespace: Namespace,
        values: Union[str, Sequence[Any], None],
        option_string: Optional[str] = None,  # noqa: ARG002
    ) -> None:
        assert not isinstance(values, str)
        assert values is not None

        parsed_dictionary: dict[Any, Any] = {}
        for key_value_string in values:
            assert isinstance(key_value_string, str)

            key_value: list[str] = key_value_string.split("=", 1)
            if len(key_value) != 2:
                msg = "Values must be given in the form 'KEY=VALUE'"
                raise ValueError(msg)

            key: Any = self.key_type(key_value[0]) if self.key_type is not None else key_value[0]
            value: Any = self.value_type(key_value[1]) if self.value_type is not None else key_value[1]
            parsed_dictionary[key] = value

        setattr(namespace, self.dest, parsed_dictionary)


def none_or_str(value: str) -> Optional[str]:
    return None if value == "None" else value


def none_or_path(value: str) -> Optional[Path]:
    return None if value == "None" else Path(value)
