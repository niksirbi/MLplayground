from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("MLplayground")
except PackageNotFoundError:
    # package is not installed
    pass
