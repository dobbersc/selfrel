import contextlib
import gc
from collections.abc import Iterator


@contextlib.contextmanager
def disable_garbage_collector() -> Iterator[None]:
    """Contextmanager to disable Python's generational garbage collection."""
    if gc.isenabled():
        gc.disable()
        try:
            yield
        finally:
            gc.enable()
    else:
        yield
