from collections.abc import Iterator

import pytest


@pytest.fixture(scope="session")
def _init_ray() -> Iterator[None]:
    import ray

    ray.init(log_to_driver=False)
    yield
    ray.shutdown()
