"""NOOP test
"""

from packagename import noop


def test_noop():
    """test src code is found properly.
    """
    assert noop() is None
