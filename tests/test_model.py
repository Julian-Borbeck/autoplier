from autoplier.model import autoPLIER
import pandas as pd
import numpy as np

# Grab the test data
X = pd.read_csv("test_data/test_X.csv", sep = ",", index_col=0)
pathways = pd.read_csv("test_data/test_pathways.csv", sep = ",", index_col=0)


# TODO: Need to update tests with expected output (number of
#  dimensions, expected values, etc).


def test_embed_basic():
    """
    Tests the simple embedding of a user dataset

    return: A pandas Dataframe with the Z embedded data
    """
    ap = autoPLIER(n_components=100)
    Z = ap.fit_transform(X, pathways, maxepoch = 100, verbose = 0)
    assert Z.__class__.__name__ == "DataFrame"


def test_embed_xy():
    """
    Tests ability to embed a Y matrix into an X-trained space

    return: An pandas Dataframe with the Y embedding data
    """
    ap = autoPLIER(n_components=100)
    ap.fit(X[0:50], pathways, maxepoch = 100, verbose = 0)
    ap.build_encoder()
    Y_embed = ap.transform(X[50:100], pathways)
    assert Y_embed.__class__.__name__ == "DataFrame"


def test_get_u_matrix():
    """
    Tests the ability to return the U matrix after fit

    return: a DataFrame with the U matrix
    """
    ap = autoPLIER(n_components=100)
    ap.fit_transform(X, pathways, maxepoch = 100, verbose = 0)
    U = ap.components_decomposition_
    assert U.__class__.__name__ == "DataFrame"
