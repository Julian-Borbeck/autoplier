from autoplier.model import autoPLIER
import numpy as np
import pandas as pd

# Grab the test data
X = pd.read_csv("test_data/test_X.csv", sep = ",")
pathways = pd.read_csv("test_data/test_pathways.csv", sep = ",")

# TODO: Need to update tests with expected output (number of
#  dimensions, expected values, etc).


def test_embed_basic():
    """
    Tests the simple embedding of a user dataset

    return: An ND-array with the Z embedded data
    """
    ap = autoPLIER(n_components=100)
    Z = ap.fit_transform(X)
    assert Z.__class__.__name__ == "ndarray"


def test_embed_xy():
    """
    Tests ability to embed a Y matrix into an X-trained space

    return: An ND-array with the Y embedding data
    """
    ap = autoPLIER(n_components=100)
    ap.fit(X[0:50])
    ap.build_encoder()
    Y_embed = ap.transform(X[50:100])
    assert Y_embed.__class__.__name__ == "ndarray"


def test_get_u_matrix():
    """
    Tests the ability to return the U matrix after fit

    return: a DataFrame with the U matrix
    """
    ap = autoPLIER(n_components=100)
    ap.fit_transform(X)
    U = ap.components_decomposition_
    assert U.__class__.__name__ == "DataFrame"
