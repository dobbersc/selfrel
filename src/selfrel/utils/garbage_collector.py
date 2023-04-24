import contextlib
import gc
from collections.abc import Iterator


@contextlib.contextmanager
def disable_garbage_collector() -> Iterator[None]:
    if gc.isenabled():
        gc.disable()
        yield
        gc.enable()
