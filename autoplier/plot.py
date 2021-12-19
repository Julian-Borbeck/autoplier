import pandas as pd
import matplotlib.pyplot as plt
from autoplier import model

def plot_topLVs(sample_df, n_LVs, figure_size):
    LV_dict = model.get_top_LVs(sample_df, n_LVs)
    df = pd.DataFrame(LV_dict).fillna(0)
    ax = df.plot.bar(rot=0, figsize=figure_size)
    ax.set_xlabel("LVs")
    return ax


def plot_top_pathways(LVs, n_pathways, figure_size):
    pathwaydict = model.get_top_pathways(LVs, n_pathways)
    df = pd.DataFrame(pathwaydict).fillna(0)
    ax = df.plot.barh(rot=0, figsize=figure_size)
    return ax

def plot_top_pathway_LVs(pathway, n_LVs, figuresize):
    LVs = model.get_top_pathway_LVs(pathway, n_LVs)
    LVs.plot.bar(figsize = figuresize)
    return LVs