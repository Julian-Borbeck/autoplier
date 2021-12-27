"""NOOP test
"""

from autoplier import noop


def test_noop():
    """test autoplier code is found properly.
    """
    assert noop() is None
