"""NOOP test
"""

from src import noop


def test_noop():
    """test autoplier code is found properly.
    """
    assert noop() is None
