import pandas as pd
import matplotlib.pyplot as plt
from autoplier import model as mod
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback

def plot_topLVs(sample_df, n_LVs, figure_size):
    LV_dict = mod.get_top_LVs(sample_df, n_LVs)
    df = pd.DataFrame(LV_dict).fillna(0)
    ax = df.plot.bar(rot=0, figsize=figure_size)
    ax.set_xlabel("LVs")
    return ax


def plot_top_pathways(LVs, n_pathways, figure_size, model):
    pathwaydict = model.get_top_pathways(LVs, n_pathways)
    df = pd.DataFrame(pathwaydict).fillna(0)
    ax = df.plot.barh(rot=0, figsize=figure_size)
    return ax

def plot_top_pathway_LVs(pathway, n_LVs, figuresize, model):
    LVs = model.get_top_pathway_LVs(pathway, n_LVs)
    ax = LVs.plot.bar(figsize = figuresize)
    return ax

def plot_learning_curve(history):
    plt.plot(history['loss'], label='loss_train')
    plt.plot(history['val_loss'], label='loss_test')

def plot_LV_range(range, X_train, X_test, y_train, y_test, pathways, regval = 1.20E-7,learning_rate= 0.000156):

    # Autoplier callbacks
    callbacks = [
        # early stopping - to mitigate overfitting
        EarlyStopping(patience=100, monitor='val_loss'),
    ]

    fscores = []
    aps = []

    for LV in range:

        model = mod.autoPLIER(regval=regval, n_components=LV, learning_rate=learning_rate)
        model.fit(X_train, pathways, callbacks, batch_size=None)

        Z_train = model.transform(X_train, pathways)
        Z_test = model.transform(X_test, pathways)

        f, ap = mod.train_classifiers(Z_train, Z_test, y_train, y_test)

        fscores += [f]
        aps += [ap]

    plt.plot(range, aps, label='Average Precision')
    plt.plot(range, fscores, label='Fscore')

