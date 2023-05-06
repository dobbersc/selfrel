from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("selfrel")
except PackageNotFoundError:
    __version__ = "unknown version"
