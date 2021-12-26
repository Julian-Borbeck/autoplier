from autoplier.plot import plot_topLVs
from autoplier.plot import plot_top_pathways
from autoplier.plot import plot_top_pathway_LVs
from autoplier.model import autoPLIER
from pathlib import Path
import pandas as pd

# TODO: Should this return an Axes object or something else?
# TODO: What other kinds of plots should we test here?

# Grab the test data
X = pd.read_csv(Path(__file__).parent  /"test_data/test_X.csv", sep=",", index_col=0)
pathways = pd.read_csv(Path(__file__).parent /"test_data/test_pathways.csv", sep=",", index_col=0)

ap = autoPLIER(n_components=100)
Z = ap.fit_transform(X, pathways, maxepoch = 100, verbose = 0)

def test_plot_topLVs():
    """
    Tests the ability to plot the U matrix as a heatmap

    return: An Axes object from matplotlib
    """
    ax = plot_topLVs(X, n_LVs = 10, figure_size= (1,1))
    assert ax.__class__.name == "AxesSubplot"

def test_plot_top_pathways():
    """
    Tests the ability to plot the U matrix as a heatmap

    return: An Axes object from matplotlib
    """
    ax = plot_top_pathways([0], n_pathways = 10, figure_size = (1,1), model = ap)
    assert ax.__class__.name == "AxesSubplot"

def test_plot_top_pathway_LVs():
    """
    Tests the ability to plot the U matrix as a heatmap

    return: An Axes object from matplotlib
    """
    ax = plot_top_pathway_LVs("BIOCARTA_CB1R_PATHWAY", n_LVs = 10, figuresize= (1,1), model= ap)
    assert ax.__class__.name == "AxesSubplot"