from collections.abc import Iterator

import pytest


@pytest.fixture(scope="session")
def init_ray() -> Iterator[None]:
    import ray

    ray.init()
    yield
    ray.shutdown()
