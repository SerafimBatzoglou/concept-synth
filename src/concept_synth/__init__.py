"""Public Concept Synth benchmark package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("concept-synth")
except PackageNotFoundError:
    __version__ = "0+local"

