from importlib.metadata import version

import molag
import molag.utils


def test_utility_exports_are_complete() -> None:
    assert set(molag.utils.__all__) == {
        "GeometryUtils",
        "preferred_device",
        "resolve_device",
    }


def test_package_version_matches_distribution_metadata() -> None:
    assert molag.__version__ == version("molag")
