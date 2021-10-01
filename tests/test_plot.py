from packagename.plot import plot_components_decomposition
from pickle import load

# Test data load
with open('tests/test_data/ap_example.pkl', 'rb') as f:
    ap_example = load(f)

# TODO: Should this return an Axes object or something else?
# TODO: What other kinds of plots should we test here?


def test_plot_components_decompoisition():
    """
    Tests the ability to plot the U matrix as a heatmap

    return: An Axes object from matplotlib
    """
    ax = plot_components_decomposition(ap_example)
    assert ax.__class__.name == "AxesSubplot"
