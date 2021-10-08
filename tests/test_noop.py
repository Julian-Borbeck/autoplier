"""NOOP test
"""

from src import noop


def test_noop():
    """test src code is found properly.
    """
    assert noop() is None
