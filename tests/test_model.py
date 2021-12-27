from autoplier.model import autoPLIER
import pandas as pd
import numpy as np
from pathlib import Path
import autoplier.model as mod

# TODO: Need to update tests with expected output (number of
#  dimensions, expected values, etc).

# Grab the test data
X = pd.read_csv(Path(__file__).parent  /"test_data/test_X.csv", sep=",", index_col=0)
pathways = pd.read_csv(Path(__file__).parent /"test_data/test_pathways.csv", sep=",", index_col=0)


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


def test_get_top_LVs():
    """
    Tests the ability to return top 10 LVs

    return: a dictionary
    """
    ap = autoPLIER(n_components=100)
    ap.fit_transform(X, pathways, maxepoch = 100, verbose = 0)
    top_LVs = mod.get_top_LVs()
    assert top_LVs.__class__.__name__ == "dict"


def test_set_seed():
    """
    Tests the ability to set seed
    """
    mod.set_seed_(111)

def test_get_top_pathways():
    """
    Tests the ability to return the 10 largest pathways for LV 0

    return: a dictionary
    """
    ap = autoPLIER(n_components=100)
    ap.fit_transform(X, pathways, maxepoch=100, verbose=0)
    top_pathways = ap.get_top_pathways([0], n_pathways = 10)
    assert top_pathways.__class__.__name__ == "dict"

def test_get_top_pathway_LVs():
    """
    Tests the ability to return the top 10 LVs in which the pathway BIOCARTA_CB1R_PATHWAY is weighted largest

    return: a Pandas Series
    """
    ap = autoPLIER(n_components=100)
    ap.fit_transform(X, pathways, maxepoch=100, verbose=0)
    top_LVs = ap.get_top_pathway_LVs("BIOCARTA_CB1R_PATHWAY", n_LVs = 10)
    assert top_LVs.__class__.__name__ == "Series"

def test_epsilon_sparsity():
    """
    Tests the epsilon sparsity measure

    return: a float
    """
    ap = autoPLIER(n_components=100)
    Z = ap.fit_transform(X, pathways, maxepoch=100, verbose=0)
    sparsity = mod.sparsity_epsilon(Z, 1.0E-4)
    assert sparsity.__class__.__name__ == "float"

def test_optimize_l1():
    """
    Tests the ability to automatically find an lv which leads to a specified sparsity

    return: a float
    """
    closest_l1 = mod.optimize_l1(0.7, 0.4, 1.0E-10, X, pathways)
    assert closest_l1.__class__.__name__ == "float"