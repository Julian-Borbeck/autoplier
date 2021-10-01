from packagename.model import autoPLIER
from pickle import load
import pandas as pd

# Grab the test data
X = pd.read_csv('tests/test_data/traindataX.csv').values()
Y = pd.read_csv('tests/test_data/traindataY.csv').values()
with open('tests/test_data/ap_example.pkl', 'rb') as f:
    ap_example = load(f)

# TODO: Need to update tests with expected output (number of
#  dimensions, expected values, etc).


def test_embed_basic():
    """
    Tests the simple embedding of a user dataset

    return: An ND-array with the X embedding data
    """
    ap = autoPLIER(n_components=100)
    X_embed = ap.fit_transform(X)
    assert X_embed.__class__.__name__ == "ndarray"


def test_embed_xy():
    """
    Tests ability to embed a Y matrix into an X-trained space

    return: An ND-array with the Y embedding data
    """
    ap = autoPLIER(n_components=100)
    ap.fit(X)
    Y_embed = ap.transform(Y)
    assert Y_embed.__class__.__name__ == "ndarray"


def test_get_z_matrix():
    """
    Tests the ability to return the Z matrix after fit

    return: a DataFrame with the Z matrix
    """
    Z = ap_example.components_
    assert Z.__class__.__name__ == "DataFrame"


def test_get_u_matrix():
    """
    Tests the ability to return the U matrix after fit

    return: a DataFrame with the U matrix
    """
    U = ap_example.components_decomposition_
    assert U.__class__.__name__ == "DataFrame"
