from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("selfrel")
except PackageNotFoundError:
    __version__ = "unknown version"
