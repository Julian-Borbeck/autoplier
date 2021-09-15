"""NOOP test
"""


from src.core import noop


def test_noop():
    """test src code is found properly.
    """
    assert(noop() is None)
