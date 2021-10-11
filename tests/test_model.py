from autoplier.model import autoPLIER
import numpy as np
import pandas as pd

# Grab the test data
X = np.load("testdata.npy", mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
pathways = pd.DataFrame()## TODO: create example pathway file
index = list(range(0, X.shape[0]))

# TODO: Need to update tests with expected output (number of
#  dimensions, expected values, etc).


def test_embed_basic():
    """
    Tests the simple embedding of a user dataset

    return: An ND-array with the Z embedded data
    """
    ap = autoPLIER(X.shape[1], n_components=100)
    Z = ap.fit_transform(X, index = index)
    assert Z.__class__.__name__ == "ndarray"


def test_embed_xy():
    """
    Tests ability to embed a Y matrix into an X-trained space

    return: An ND-array with the Y embedding data
    """
    ap = autoPLIER(X.shape[1], n_components=100)
    ap.fit(X[0:50])
    ap.build_encoder()
    Y_embed = ap.transform(X[50:100], index = index[50:100])
    assert Y_embed.__class__.__name__ == "ndarray"


def test_get_u_matrix():
    """
    Tests the ability to return the U matrix after fit

    return: a DataFrame with the U matrix
    """
    ap = autoPLIER(X.shape[1], n_components=100)
    ap.fit_transform(X, index=index)
    U = ap.components_decomposition_
    assert U.__class__.__name__ == "DataFrame"
