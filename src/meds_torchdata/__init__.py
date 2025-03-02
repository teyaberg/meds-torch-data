from importlib.metadata import PackageNotFoundError, version

__package_name__ = "meds_torchdata"

try:
    __version__ = version(__package_name__)
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "unknown"
