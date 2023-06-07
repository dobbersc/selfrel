from collections.abc import Iterator
from typing import TypeVar

TypeT = TypeVar("TypeT", bound=type)


def iter_subclasses(cls: TypeT) -> Iterator[TypeT]:
    subclasses = cls.__subclasses__()
    yield from subclasses
    for subclass in subclasses:
        yield from iter_subclasses(subclass)
